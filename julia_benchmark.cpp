#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>
#include <atomic>
#include <chrono>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

using namespace std;

struct Complex {
    double r;
    double i;

    inline double magnitude2() const { return r * r + i * i; }
}; // Complex

struct Image {
    int width;
    int height;
    vector<float> data;

    Image(int w, int h) : width(w), height(h), data(width * height, 0) {}

    void clear() { fill(data.begin(), data.end(), 0.0); } // clear
}; // Image

struct Colour { unsigned char r, g, b; }; // Colour

Colour palette(float t) {
    Colour c;
    double invt = 1 - t;
    c.r = (unsigned char) (255 * (9 * invt * t * t * t));
    c.g = (unsigned char) (255 * (15 * invt * invt * t * t));
    c.b = (unsigned char) (255 * (8.5 * invt * invt * invt * t));
    return c;
}

struct JuliaRenderer {
    const int width = 6400;
    const int height = 4000;
    const int maxIter = 1000;

    const double xmin = -1.8;
    const double xmax = 1.8;
    const double ymin = -1.2;
    const double ymax = 1.2;

//    const int tile = 64;
//    const int rowSkip = max(1, tile * tile / width);

    const Complex c = {-0.7, 0.256};

    Image img;

    JuliaRenderer() : img(width, height) {}

    inline float escapeTest(Complex& z) {
	int n = 0;
	while (n < maxIter) {
	    double zr2 = z.r * z.r;
	    double zi2 = z.i * z.i;
	    double zri = z.r * z.i;
	    if (zr2 + zi2 > 4.0) break;

	    z.i = 2.0 * zri + c.i;
	    z.r = zr2 - zi2 + c.r;
	    n++;
	}
	return n - log2(log(z.magnitude2()) / 2); // Calculations are performed as doubles, floats are stored
    } // escapeTest

    inline void pixelToComplex(int px, int py, Complex& z) {
	z.r = xmin + (xmax - xmin) * px / width;
	z.i = ymin + (ymax - ymin) * py / height;
    } // pixelToComplex

    void simpleRender() {
	auto startTime = chrono::high_resolution_clock::now();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Complex z;
		pixelToComplex(x, y, z);
		float val = escapeTest(z);
		img.data[y * width + x] = val;
            }
        }
	auto endTime = chrono::high_resolution_clock::now();
	float duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	cout << "Time taken is " << duration / 1000 << " seconds (Simple)." << endl;

    } // simpleRender

    void renderRow(int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            for (int x = 0; x < width; x++) {
                Complex z;
		pixelToComplex(x, y, z);
		float val = escapeTest(z);
		img.data[y * width + x] = val;
            }
        }
    } // renderRow

    void renderTile(int x0, int x1, int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Complex z;
		pixelToComplex(x, y, z);
		float val = escapeTest(z);
		img.data[y * width + x] = val;
            }
        }
    } // renderTile

    void renderParallelRows() {
	int threadNum = thread::hardware_concurrency();
	// cout << "There are " << threadNum << " possible concurrent threads." << endl;

	int rowSkip = max(1, height / (3 * threadNum));

	vector<thread> workers;
	workers.reserve(threadNum);

	atomic<int> nextRow = 0;
	auto worker = [&]() {
	    while (true) {
		int y0 = nextRow.fetch_add(rowSkip, memory_order_relaxed);
		if (y0 >= height) return;
		int y1 = min(y0 + rowSkip, height);
		renderRow(y0, y1);
	    }
	};

	auto startTime = chrono::high_resolution_clock::now();
	for (int t = 0; t < threadNum; ++t) workers.emplace_back(worker);
	for (thread& w : workers) w.join();
	auto endTime = chrono::high_resolution_clock::now();
	float duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	cout << "Time taken is " << duration / 1000 << " seconds (Rows)." << endl;
    }

    void renderParallelTiles() {
	int threadNum = thread::hardware_concurrency();
	// cout << "There are " << threadNum << " possible concurrent threads." << endl;

	int tile = 64;

	vector<thread> workers;
	workers.reserve(threadNum);

	atomic<int> nextTile = 0;
	int wcount = (width + tile - 1) / tile;
	auto worker = [&]() {
	    while (true) {
		int id = nextTile.fetch_add(1, memory_order_relaxed);
		int x0 = tile * (id % wcount);
		int y0 = tile * (id / wcount);
		if (y0 >= height) return;
		int x1 = min(x0 + tile, width);
		int y1 = min(y0 + tile, height);
		renderTile(x0, x1, y0, y1);
	    }
	};

	auto startTime = chrono::high_resolution_clock::now();
	for (int t = 0; t < threadNum; ++t) workers.emplace_back(worker);
	for (thread& w : workers) w.join();
	auto endTime = chrono::high_resolution_clock::now();
	float duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	cout << "Time taken is " << duration / 1000 << " seconds (Tiles)." << endl;
    }

    void renderParallelRowsStatic() {
	int rowSkip = 1;

	vector<thread> workers;

	auto startTime = chrono::high_resolution_clock::now();
	for (int y0 = 0; y0 < height; y0 += rowSkip) {
	    int y1 = min(y0 + rowSkip, height);
	    workers.emplace_back(&JuliaRenderer::renderRow, this, y0, y1);
	}
	for (thread& w : workers) w.join();
	auto endTime = chrono::high_resolution_clock::now();
	float duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	cout << "Time taken is " << duration / 1000 << " seconds (Static Rows)." << endl;
    }

    void renderParallelTilesStatic() {
	int tile = 64;

	vector<thread> workers;

	auto startTime = chrono::high_resolution_clock::now();
	for (int y0 = 0; y0 < height; y0 += tile) {
	    int y1 = min(y0 + tile, height);
	    for (int x0 = 0; x0 < width; x0 += tile) {
		int x1 = min(x0 + tile, width);
		workers.emplace_back(&JuliaRenderer::renderTile, this, x0, x1, y0, y1);
	    }
	}
	for (thread& w : workers) w.join();
	auto endTime = chrono::high_resolution_clock::now();
	float duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	cout << "Time taken is " << duration / 1000 << " seconds (Static Tiles)." << endl;
    }

    void saveImg() {
	for (int y = 0; y < height; y++) {
	    for (int x = 0; x < width; x++) {
		float v = img.data[y * width + x];
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
}; // JuliaRenderer

int main() {
    JuliaRenderer jr;
    jr.renderParallelRows();
    jr.img.clear();
    jr.renderParallelTiles();
    jr.img.clear();
    jr.renderParallelRowsStatic();
    jr.img.clear();
    jr.renderParallelTilesStatic();
    jr.img.clear();
    jr.simpleRender();
    jr.saveImg();

    cout << "Image rendered and saved" << endl;

    return 0;
} // main

