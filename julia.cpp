#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <chrono>

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
    const int width = 25600;
    const int height = 16000;
    const int maxIter = 1000;

    const double xmin = -1.8;
    const double xmax = 1.8;
    const double ymin = -1.2;
    const double ymax = 1.2;

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

    void renderParallel() {
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
	cout << "Time taken is " << duration / 1000 << " seconds with size " << tile << " tiles." << endl;
    }

    void saveImg() {
        ofstream out("julia.ppm", ios::binary);

	out << "P6\n"
	    << width << " "
	    << height << "\n255\n";

	for (int y = 0; y < height; y++) {
	    for (int x = 0; x < width; x++) {
	        float v = img.data[y * width + x];
		Colour c;
		float t = v / maxIter;
		c = palette(t);

		out.put(c.r);
		out.put(c.g);
		out.put(c.b);
	    }
	}
    } // saveImg
}; // JuliaRenderer

int main() {
    JuliaRenderer jr;
    jr.renderParallel();
    jr.saveImg();

    cout << "Image rendered and saved" << endl;

    return 0;
} // main

