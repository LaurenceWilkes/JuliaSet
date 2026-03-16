## Julia Set Renderer

This project renders a Julia set using three different approaches:

- Plain CPU implementation
- Multithreaded CPU implementation
- GPU implementation using OpenCL

### Performance

Running `julia.cpp` on the test configuration produced the following timings:

| Method | Time |
|------|------|
| Plain CPU | 33.1631 s |
| Multithreaded CPU | 4.68771 s |
| GPU (OpenCL) | 0.193511 s |

test config:
```
const int width = 6400; 
const int height = 4000;
const int maxIter = 1000;

const float xmin = -1.8;
const float xmax = 1.8;
const float ymin = -1.2;
const float ymax = 1.2;

const float cr = -0.7f;
const float ci = 0.256f;
```

---

## Dependencies

This project uses **stb_image_write** to export the generated image.

Repository:  
https://github.com/nothings/stb

The header is vendored in:

```
third_party/stb/
```

---

## Compilation

For my future reference, I compile with:

```
g++ julia.cpp -O2 -L. -lOpenCL -pthread -o julia
```

due to libOpenCL.a being in the directory

