## Parellel Image Segmentation

This is a collection of a couple parallel image segmentation algorithms. These will find the best 
way to partitions the given image in two parts: a monochromatic rectangle and a monochromatic 
background by minimizing the sum of squared errors.

### Interface function

```
Result segment(in ny, int nx, const float* data)
```

The input is given by **ny \* nx** pixel color image **data**, where each pixel constist of three 
color components, red, green and blue. Thus, there are **ny \* nx \* 3** floating point numbers 
in the array **data**.

The color components are numbered **0 <= c < 3**, **x** coordinates are numbered **0 <= x < nx**, 
**y** coordinates are numbered **0 <= y < ny**, and the value of this color component is stored in 
**data[c + 3 \* x + 3 \* nx \* y]**.

### Output

```
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};
```

In the structure above, the first four fields indicate the location of the rectangle. The upper left 
corner of the rectangle is at coordinates (x0, y0), and the lower right corner is at coordinates 
(x1-1, y1-1). That is, the width of the rectangle is x1-x0 pixels and the height is y1-y0 pixels. The 
coordinates have to satisfy 0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx.

The last two fields indicate the color of the background and the rectangle. Field outer contains the 
three color components of the background and field inner contains the three color components of the rectangle.

### The algorithm

For each pixel (x,y) and color component c, we define the error(y,x,c) as follows:

- Let color(y,x,c) = data[c + 3 * x + 3 * nx * y].
- If (x,y) is located outside the rectangle: error(y,x,c) = outer[c] - color(y,x,c).
- If (x,y) is located inside the rectangle: error(y,x,c) = inner[c] - color(y,x,c).

The total cost of the segmentation is the sum of squared errors, that is, the sum of error(y,x,c) * error(y,x,c) 
over all 0 <= c < 3 and 0 <= x < nx and 0 <= y < ny.

The algoritm finds a segmentation that minimizes the total cost.

### Versions

There are two different versions. Both are multithredded with omp and take advantage of AVX-512 vector registers.

- [Segmantation with color images, AVX-512, double precision](./src/is_avx512) 
200 x 200: 0.89 s  
400 x 400: 1.53 s

- [Segmantation with monochromatic images, AVX-512, single precision](./src/is_avx512_monochrome)  
Each pixel in the input image is assumed to have a RGB value of either (1,1,1) (white) or (0,0,0) (black).  
200 x 200: 0.016 s  
400 x 400: 0.17 s  
600 x 600: 0.84 s

The benchmarks were done on a machine with following specs:

- Intel Xeon W-2255, 10 cores, 20 threads, 3.70 / 4.50 GHz
- 64GB DDR4 (4 x 16)
- Nvidia Quadro RTX 4000
