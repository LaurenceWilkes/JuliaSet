#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

using namespace std;

struct Parameters {
    const int width = 6400;    // Pixel width
    const int height = 4000;   // Pixel height
    const int maxIter = 1000;  // Iterations of escape test

    const float xmin = -1.8;   // Frame bounds
    const float xmax = 1.8;    // x - real axis
    const float ymin = -1.2;   // y - imaginary
    const float ymax = 1.2;    //

    const int trials = 3;      // Benchmarking trials

    const float cr = -0.7f;    // Complex constant defining
    const float ci = 0.256f;   // the Julia set
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

struct JuliaRenderer {
    Parameters p;

    Image img;
    vector<unsigned char> pixels;

    JuliaRenderer() : img(p.width, p.height), pixels(p.width * p.height * 3) {}

    inline float escapeTest(float zr, float zi) {
	int n = 0;
	while (n < p.maxIter) {
	    float zr2 = zr * zr;
	    float zi2 = zi * zi;
	    float zri = zr * zi;
	    if (zr2 + zi2 > 4.0) break;

	    zi = 2.0 * zri + p.ci;
	    zr = zr2 - zi2 + p.cr;
	    n++;
	}
	if (n == p.maxIter) return p.maxIter;
	return n - log2(log(zr * zr + zi * zi) / 2);
    } // escapeTest

    inline void pixelToComplex(int px, int py, float& zr, float& zi) {
	zr = p.xmin + (p.xmax - p.xmin) * px / (float) p.width;
	zi = p.ymin + (p.ymax - p.ymin) * py / (float) p.height;
    } // pixelToComplex

    double renderPlain() {
	auto startTime = chrono::high_resolution_clock::now();
	for (int y = 0; y < p.height; y++) {
	    int yw = y * p.width;
	    for (int x = 0; x < p.width; x++) {
		float zr, zi;
		pixelToComplex(x, y, zr, zi);
		img.contents[yw + x] = escapeTest(zr, zi);
	    }
	}
	auto endTime = chrono::high_resolution_clock::now();
	double duration = chrono::duration<double>(endTime - startTime).count();
	return duration;
    } // renderPlain

    void renderTile(int x0, int x1, int y0, int y1) {
        for (int y = y0; y < y1; y++) {
	    int yw = y * p.width;
            for (int x = x0; x < x1; x++) {
		float zr, zi;
		pixelToComplex(x, y, zr, zi);
		img.contents[yw + x] = escapeTest(zr, zi);
            }
        }
    } // renderTile

    double renderThreads() {
	int threadNum = thread::hardware_concurrency();
	// cout << "There are " << threadNum << " possible concurrent threads." << endl;

	int tile = 64;

	vector<thread> workers;
	workers.reserve(threadNum);

	atomic<int> nextTile = 0;
	int wcount = (p.width + tile - 1) / tile;
	auto worker = [&]() {
	    while (true) {
		int id = nextTile.fetch_add(1, memory_order_relaxed);
		int x0 = tile * (id % wcount);
		int y0 = tile * (id / wcount);
		if (y0 >= p.height) return;
		int x1 = min(x0 + tile, p.width);
		int y1 = min(y0 + tile, p.height);
		renderTile(x0, x1, y0, y1);
	    }
	};

	auto startTime = chrono::high_resolution_clock::now();
	for (int t = 0; t < threadNum; ++t) workers.emplace_back(worker);
	for (thread& w : workers) w.join();
	auto endTime = chrono::high_resolution_clock::now();
	double duration = chrono::duration<double>(endTime - startTime).count();
	return duration;
    }

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
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &p.cr);
        err |= clSetKernelArg(kernel, arg++, sizeof(float), &p.ci);
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

	int success = stbi_write_png("julia.png", p.width, p.height, 3, pixels.data(), p.width * 3);

	if (!success) { cerr << "Failed to write PNG\n"; }
    }
}; // JuliaRenderer

int main() {
    Parameters p;

    JuliaRenderer jr;

    cout << "Initialising...";
    jr.renderPlain(); jr.img.clear();
    jr.renderThreads(); jr.img.clear();
    jr.renderGPU(); jr.img.clear();
    cout << "done\n";

    double timePlain = 0, timeThreads = 0, timeGPU = 0;
    for (int i = 0; i < p.trials; i++) {
	timePlain += jr.renderPlain();
	jr.img.clear();
	timeThreads += jr.renderThreads();
	jr.img.clear();
	timeGPU += jr.renderGPU();
	jr.img.clear();
    }
    cout << "Time for plain algorithm: " << timePlain / p.trials << " seconds.\n";
    cout << "Time for multithreaded algorithm: " << timeThreads / p.trials << " seconds.\n";
    cout << "Time for GPU-based algorithm: " << timeGPU / p.trials << " seconds.\n";

    jr.renderGPU(); // Render image properly
    jr.saveImg();   //
    cout << "Image rendered and saved" << endl;

    return 0;
} // main

