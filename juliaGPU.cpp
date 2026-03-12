#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <windows.h>
#include <chrono>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

using namespace std;

#define LOAD(name) name = (decltype(name))GetProcAddress(lib, #name + 2)

decltype(clGetPlatformIDs)* p_clGetPlatformIDs;
decltype(clGetDeviceIDs)* p_clGetDeviceIDs;
decltype(clCreateContext)* p_clCreateContext;
decltype(clCreateCommandQueue)* p_clCreateCommandQueue;
decltype(clCreateProgramWithSource)* p_clCreateProgramWithSource;
decltype(clBuildProgram)* p_clBuildProgram;
decltype(clCreateKernel)* p_clCreateKernel;
decltype(clSetKernelArg)* p_clSetKernelArg;
decltype(clCreateBuffer)* p_clCreateBuffer;
decltype(clEnqueueNDRangeKernel)* p_clEnqueueNDRangeKernel;
decltype(clEnqueueReadBuffer)* p_clEnqueueReadBuffer;
decltype(clFinish)* p_clFinish;
decltype(clReleaseMemObject)* p_clReleaseMemObject;
decltype(clReleaseKernel)* p_clReleaseKernel;
decltype(clReleaseProgram)* p_clReleaseProgram;
decltype(clReleaseCommandQueue)* p_clReleaseCommandQueue;
decltype(clReleaseContext)* p_clReleaseContext;
decltype(clGetProgramBuildInfo)* p_clGetProgramBuildInfo;

void loadOpenCL() {
    HMODULE lib = LoadLibraryA("OpenCL.dll");

    if (!lib) {
        cerr << "OpenCL.dll not found\n";
        exit(1);
    }

    LOAD(p_clGetPlatformIDs);
    LOAD(p_clGetDeviceIDs);
    LOAD(p_clCreateContext);
    LOAD(p_clCreateCommandQueue);
    LOAD(p_clCreateProgramWithSource);
    LOAD(p_clBuildProgram);
    LOAD(p_clCreateKernel);
    LOAD(p_clSetKernelArg);
    LOAD(p_clCreateBuffer);
    LOAD(p_clEnqueueNDRangeKernel);
    LOAD(p_clEnqueueReadBuffer);
    LOAD(p_clFinish);
    LOAD(p_clReleaseMemObject);
    LOAD(p_clReleaseKernel);
    LOAD(p_clReleaseProgram);
    LOAD(p_clReleaseCommandQueue);
    LOAD(p_clReleaseContext);
    LOAD(p_clGetProgramBuildInfo);

    if (!p_clGetPlatformIDs) {
        cerr << "Failed to load OpenCL functions\n";
        exit(1);
    }
}

struct Colour { unsigned char r, g, b; };

Colour palette(float t) {
    Colour c;
    float invt = 1.0f - t;

    c.r = (unsigned char)(255 * (9 * invt * t * t * t));
    c.g = (unsigned char)(255 * (15 * invt * invt * t * t));
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

    vector<float> data;
    vector<unsigned char> pixels;

    JuliaRenderer() : data(width * height), pixels(width * height * 3) {}

    string loadKernel() {
        ifstream f("julia_kernel.cl");

        if (!f) {
            cerr << "Kernel file julia_kernel.cl not found\n";
            exit(1);
        }

        stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    void render() {
        loadOpenCL();

        cl_platform_id platform;
        cl_device_id device;

        cl_int err;

        err = p_clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "No OpenCL platform found\n";
            exit(1);
        }

        err = p_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "No GPU OpenCL device found\n";
            exit(1);
        }

        cl_context context = p_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

        cl_command_queue queue = p_clCreateCommandQueue(context, device, 0, &err);

        string src = loadKernel();
        const char* csrc = src.c_str();
        size_t len = src.size();

        cl_program program =
            p_clCreateProgramWithSource(context, 1, &csrc, &len, &err);

        err = p_clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

        if (err != CL_SUCCESS) {

            size_t logSize;

            p_clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

            vector<char> log(logSize);

            p_clGetProgramBuildInfo(program, device,
                CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

            cerr << "Kernel build error:\n" << log.data() << endl;

            exit(1);
        }

        cl_kernel kernel = p_clCreateKernel(program, "julia", &err);

        size_t imgSize = width * height * sizeof(float);

        cl_mem outBuffer =
            p_clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

        int arg = 0;

        p_clSetKernelArg(kernel, arg++, sizeof(cl_mem), &outBuffer);
        p_clSetKernelArg(kernel, arg++, sizeof(int), &width);
        p_clSetKernelArg(kernel, arg++, sizeof(int), &height);
        p_clSetKernelArg(kernel, arg++, sizeof(int), &maxIter);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &xmin);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &xmax);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &ymin);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &ymax);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &cr);
        p_clSetKernelArg(kernel, arg++, sizeof(float), &ci);

        size_t global[2] = {(size_t)width, (size_t)height};

        auto startTime = chrono::high_resolution_clock::now();

        p_clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            nullptr,
            global,
            nullptr,
            0,
            nullptr,
            nullptr
        );

        p_clFinish(queue);

        auto endKernel = chrono::high_resolution_clock::now();

        p_clEnqueueReadBuffer(
            queue,
            outBuffer,
            CL_TRUE,
            0,
            imgSize,
            data.data(),
            0,
            nullptr,
            nullptr
        );

        auto endTotal = chrono::high_resolution_clock::now();

        double kernelTime =
            chrono::duration<double>(endKernel - startTime).count();

        double totalTime =
            chrono::duration<double>(endTotal - startTime).count();

        cout << "Kernel time: " << kernelTime << " seconds\n";
        cout << "Total GPU render time: " << totalTime << " seconds\n";

        p_clReleaseMemObject(outBuffer);
        p_clReleaseKernel(kernel);
        p_clReleaseProgram(program);
        p_clReleaseCommandQueue(queue);
        p_clReleaseContext(context);
    }

    void save()
    {
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {

            float v = data[y * width + x];
            float t = v / maxIter;

            Colour c = palette(t);

            int idx = (y * width + x) * 3;

            pixels[idx]     = c.r;
            pixels[idx + 1] = c.g;
            pixels[idx + 2] = c.b;
        }

        stbi_write_png("juliaGPU.png", width, height, 3, pixels.data(), width * 3);
    }
};

int main()
{
    JuliaRenderer r;

    r.render();
    r.save();

    cout << "Image rendered and saved\n";
}
