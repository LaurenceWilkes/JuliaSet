#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

using namespace std;

struct Parameters {
    const int width = 16000;    // Pixel width
    const int height = 16000;   // Pixel height
    const int maxIter = 100;  // Iterations of escape test

    const float xmin = -1.8;   // Frame bounds
    const float xmax = 0.6;    //
    const float ymin = -1.2;   // x - real axis
    const float ymax = 1.2;    // y - imaginary
}; // Parameters

struct Image {
    int width;
    int height;
    vector<float> contents;

    Image(int w, int h) : width(w), height(h), contents(width * height, 0) {}

    float* data() { return contents.data(); }

    void clear() { fill(contents.begin(), contents.end(), 0.0); } // clear
}; // Image

struct Colour { unsigned char r, g, b; }; // Colour

Colour palette(float t) {
    Colour c;
    float invt = 1.0f - t;
    c.r = (unsigned char) (255 * (9.0f * invt * t * t * t));
    c.g = (unsigned char) (255 * (15.0f * invt * invt * t * t));
    c.b = (unsigned char) (255 * (8.5f * invt * invt * invt * t));
    return c;
}

struct MandelRenderer {
    Parameters p;

    Image img;
    vector<unsigned char> pixels;

    MandelRenderer() : img(p.width, p.height), pixels(p.width * p.height * 3) {}

    double renderGPU() {
        cl_int err;

        cl_platform_id platform;
        cl_device_id device;

        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) { cerr << "No OpenCL platform found\n"; exit(1); }

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) { cerr << "No GPU OpenCL device found\n"; exit(1); }

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

        ifstream f("mandelbrot_kernel.cl");
        if (!f) cerr << "Kernel file mandelbrot_kernel.cl not found\n";
        stringstream ss;
        ss << f.rdbuf();
        string src = ss.str();
        const char* csrc = src.c_str();
        size_t len = src.size();

        cl_program program = clCreateProgramWithSource(context, 1, &csrc, &len, &err);

        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) { cerr << "Kernel build error:\n" << endl; exit(1); }

        cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);

        size_t imgSize = p.width * p.height * sizeof(float);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

	const float dx = (p.xmax - p.xmin) / p.width;
	const float dy = (p.ymax - p.ymin) / p.height;
        int arg = 0;
        err  = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &outBuffer);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &p.width);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &p.height);
        err |= clSetKernelArg(kernel, arg++, sizeof(int), &p.maxIter);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &p.xmin);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &p.ymin);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &dx);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &dy);
        if (err != CL_SUCCESS) { cerr << "Arg error:\n" << endl; exit(1); }

        size_t global[2] = {(size_t) p.width, (size_t) p.height};

        auto startTime = chrono::high_resolution_clock::now(); // Start calculation

        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);

        auto endTime = chrono::high_resolution_clock::now(); // End read buffer

        double duration = chrono::duration<double>(endTime - startTime).count();

        clReleaseMemObject(outBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

	return duration;
    }

    void saveImg() {
	for (int y = 0; y < p.height; y++) {
	    int yw = y * p.width;
	    for (int x = 0; x < p.width; x++) {
		float v = img.contents[yw + x];
		float t = v / p.maxIter;
		Colour c = palette(t);
		int idx = (yw + x) * 3;
		pixels[idx]     = c.r;
		pixels[idx + 1] = c.g;
		pixels[idx + 2] = c.b;
	    }
	}

	int success = stbi_write_png("mandelbrot.png", p.width, p.height, 3, pixels.data(), p.width * 3);

	if (!success) { cerr << "Failed to write PNG\n"; }
    }
}; // MandelRenderer

int main() {
    Parameters p;

    MandelRenderer mr;

    double timeGPU = mr.renderGPU();
    cout << "Run time: " << timeGPU << " seconds.\n";

    cout << "Saving...";
    mr.saveImg();
    cout << "done\n";
    cout << "Image rendered and saved" << endl;

    return 0;
} // main

