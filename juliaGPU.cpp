#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

using namespace std;

struct Colour { unsigned char r, g, b; }; // Colour

Colour palette(float t) {
    Colour c;
    float invt = 1.0f - t;

    c.r = (unsigned char)(255 * (9.0f * invt * t * t * t));
    c.g = (unsigned char)(255 * (15.0f * invt * invt * t * t));
    c.b = (unsigned char)(255 * (8.5f * invt * invt * invt * t));

    return c;
}

struct JuliaRenderer {
    const int width = 12800;
    const int height = 8000;
    const int maxIter = 1000;

    const float xmin = -1.8f;
    const float xmax = 1.8f;
    const float ymin = -1.2f;
    const float ymax = 1.2f;

    const float cr = -0.7f;
    const float ci = 0.256f;

    vector<float> img;
    vector<unsigned char> pixels;

    JuliaRenderer() : img(width * height), pixels(width * height * 3) {}

    void render() {
        cl_int err;

        cl_platform_id platform;
        cl_device_id device;

        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) { cerr << "No OpenCL platform found\n"; exit(1); }

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) { cerr << "No GPU OpenCL device found\n"; exit(1); }

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

        ifstream f("julia_kernel.cl");
        if (!f) cerr << "Kernel file julia_kernel.cl not found\n";
        stringstream ss;
        ss << f.rdbuf();
        string src = ss.str();
        const char* csrc = src.c_str();
        size_t len = src.size();

        cl_program program = clCreateProgramWithSource(context, 1, &csrc, &len, &err);

        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) { cerr << "Kernel build error:\n" << endl; exit(1); }

        cl_kernel kernel = clCreateKernel(program, "julia", &err);

        size_t imgSize = width * height * sizeof(float);

        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

        int arg = 0;
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &outBuffer);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &width);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &height);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &maxIter);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &xmin);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &xmax);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &ymin);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &ymax);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &cr);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &ci);
        if (err != CL_SUCCESS) { cerr << "Arg error:\n" << endl; exit(1); }

        size_t global[2] = {(size_t) width, (size_t) height};

        auto startTime = chrono::high_resolution_clock::now();

        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        auto endKernel = chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);

        auto endTotal = chrono::high_resolution_clock::now();

        double kernelTime = chrono::duration<double>(endKernel - startTime).count();
        double totalTime = chrono::duration<double>(endTotal - startTime).count();

        cout << "Kernel time: " << kernelTime << " seconds\n";
        cout << "Total GPU render time: " << totalTime << " seconds\n";

        clReleaseMemObject(outBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    void saveImg() {
	for (int y = 0; y < height; y++) {
	    for (int x = 0; x < width; x++) {
		float v = img[y * width + x];
		float t = v / maxIter;
		Colour c = palette(t);
		int idx = (y * width + x) * 3;
		pixels[idx]     = c.r;
		pixels[idx + 1] = c.g;
		pixels[idx + 2] = c.b;
	    }
	}

	int success = stbi_write_png("julia.png", width, height, 3, pixels.data(), width * 3);
	if (!success) { cerr << "Failed to write PNG\n"; }
    }
};

int main() {
    JuliaRenderer r;

    r.render();
    r.saveImg();

    cout << "Image rendered and saved\n";
}
