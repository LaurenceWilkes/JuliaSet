This project uses stb_image_write from:
https://github.com/nothings/stb

The header is vendored in third_party/stb.

Compile with `g++ juliaGPU.cpp -O2 -Ithird_party/opencl -L. -lOpenCL -o juliaGPU`
