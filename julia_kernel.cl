__kernel void julia(__global float *output,
		    int width,
		    int height,
		    int maxIter,
		    float xmin,
		    float xmax,
		    float ymin,
		    float ymax,
		    float cr,
		    float ci) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float zr = xmin + (xmax - xmin) * x / width;
    float zi = ymin + (ymax - ymin) * y / height;

    int n = 0;
    while (n < maxIter) {
        float zr2 = zr * zr;
        float zi2 = zi * zi;

        if (zr2 + zi2 > 4.0f) break;

        float zri = zr * zi;

        zi = 2.0f * zri + ci;
        zr = zr2 - zi2 + cr;

        n++;
    }

    float val;
    if (n == maxIter) val = (float) maxIter;
    else val = n - log2(log(zr * zr + zi * zi) / 2.0f);

    output[y * width + x] = val;
}
