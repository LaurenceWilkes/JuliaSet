#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <fstream>
#include <algorithm>

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
    const int width = 3200;
    const int height = 2000;
    const int maxIter = 1000;

    const double xmin = -1.8;
    const double xmax = 1.8;
    const double ymin = -1.2;
    const double ymax = 1.2;

    const int tile = 64;
    const int rowSkip = tile * tile / width;

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
	return n - log2(log(z.magnitude2()) / 2); // Calculations double, floats stored
    } // escapeTest

    inline void pixelToComplex(int px, int py, Complex& z) {
	z.r = xmin + (xmax - xmin) * px / width;
	z.i = ymin + (ymax - ymin) * py / height;
    } // pixelToComplex

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

    void renderParallelTile() {
	vector<thread> workers;

	for (int ty = 0; ty < height; ty += tile) {
	    for (int tx = 0; tx < width; tx += tile) {
	        int x1 = min(tx + tile, width);
	        int y1 = min(ty + tile, height);
		workers.emplace_back(&JuliaRenderer::renderTile, this, tx, x1, ty, y1);
	    }
	}
	for (auto& t : workers) t.join();
    } // renderParallel

    void renderParallelRows() {
	vector<thread> workers;

	int ty = 0;
	while (ty < height) {
	    int y1 = min(ty + rowSkip, height);
	    workers.emplace_back(&JuliaRenderer::renderRows, this, ty, y1);
	    ty += rowSkip;
	}

	for (auto& t : workers) t.join();
    } // renderParallel

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
    jr.renderParallelTile();
    jr.saveImg();

    cout << "Image rendered and saved" << endl;

    return 0;
} // main

