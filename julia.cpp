#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <fstream>
#include <algorithm>
#include <atomic>

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
    const int width = 12800;
    const int height = 8000;
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

    void renderRows(int y0, int y1) {
        for (int y = y0; y < y1; y++) {
            for (int x = 0; x < width; x++) {
                Complex z;
		pixelToComplex(x, y, z);
		float val = escapeTest(z);
		img.data[y * width + x] = val;
            }
        }
    } // renderRows

    void renderParallelRows() {
	int threadNum = thread::hardware_concurrency();
	cout << "There are " << threadNum << " possible concurrent threads." << endl;

	int rowSkip = max(1, height / (8 * threadNum));

	vector<thread> workers;
	workers.reserve(threadNum);

	atomic<int> nextRow = 0;
	auto worker = [&]() {
	    while (true) {
		int y0 = nextRow.fetch_add(rowSkip, memory_order_relaxed);
		if (y0 >= height) return;
		int y1 = min(y0 + rowSkip, height);
		renderRows(y0, y1);
	    }
	};

	for (int t = 0; t < threadNum; ++t) workers.emplace_back(worker);
	for (thread& w : workers) w.join();
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
    jr.renderParallelRows();
    jr.saveImg();

    cout << "Image rendered and saved" << endl;

    return 0;
} // main

