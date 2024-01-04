#include <omp.h>
#include <cstdlib>

typedef double double4_t __attribute__ ((vector_size (32)));

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

inline void* aligned_malloc(std::size_t bytes) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 32, bytes)) {
        return nullptr;
    }
    return ret;
}

inline double4_t* double4_alloc(std::size_t n) {
    return static_cast<double4_t*>(aligned_malloc(sizeof(double4_t) * n));
}

Result segment(int ny, int nx, const float* data) {
    double allpx = nx * ny;
    double4_t* prec_d = double4_alloc((nx + 1) * (ny + 1));
    double4_t* vecd_d = double4_alloc(allpx);
    double best = -1;
    int x0_ret = 0, y0_ret = 0, x1_ret = 0, y1_ret = 0;

    // Vectorize data
    #pragma omp parallel
    for(int y=0; y < ny; y++) {
        for(int x = 0; x < nx; x++) {
            for(int c = 0; c < 3; c++) {
                vecd_d[x + nx * y][c] = data[c + 3 * (x + nx * y)];
            }
        }
    }
    // Precompute data for faster innerloop
    // Initialize
    for(int y=0; y < ny + 1; y++) {
        for(int x=0; x < nx + 1; x++) {
            prec_d[x + (nx + 1) * y] = (double4_t){0,0,0,0};
        }
    }
    // Calculate
    for(int y1=0; y1 < ny; y1++) {
        for(int x1=0; x1 < nx; x1++) {
            prec_d[(x1 + 1) + (nx + 1) * (y1 + 1)] =
                  vecd_d[x1 + nx * y1] 
                + prec_d[x1 + (nx + 1) * (y1 + 1)]
                + prec_d[(x1 + 1) + (nx + 1) * y1]
                - prec_d[x1 + (nx + 1) * y1];
        }
    }

    // Find the best shape and position for the rectangle
    double4_t colors_vec = prec_d[nx + (nx + 1) * ny];
    #pragma omp parallel
    {
        double inner_best = -1;

        int x0_inner = 0, y0_inner = 0, x1_inner = 0, y1_inner = 0;
        // Try all possible sizes h * w
        for(int h=1; h <= ny; h++) {
            #pragma omp for nowait schedule(dynamic, 2)
            for(int w=1; w <= nx; w++) {
                double px_x = h * w;
                double px_y = allpx - px_x;
                px_x = 1 / px_x;
                px_y = 1 / px_y;
                // Try all possible positions
                for(int y0=0; y0 <= ny - h; y0++) {
                    for(int x0=0; x0 <= nx - w; x0++) {
                        int x1 = x0 + w;
                        int y1 = y0 + h;
                        int y0_nx = (nx + 1) * y0;
                        int y1_nx = (nx + 1) * y1;
                        double current = 0;
                        double4_t sum_color_X =
                              prec_d[x1 + y1_nx]
                            - prec_d[x0 + y1_nx]
                            - prec_d[x1 + y0_nx]
                            + prec_d[x0 + y0_nx];
                        double4_t sum_color_Y = colors_vec - sum_color_X;
                        double4_t sq_add = sum_color_X * sum_color_X * px_x
                                         + sum_color_Y * sum_color_Y * px_y;
                        current = sq_add[0] + sq_add[1] + sq_add[2];
                        if(current > inner_best) {
                            inner_best = current;
                            x0_inner = x0;
                            y0_inner = y0;
                            x1_inner = x1;
                            y1_inner = y1;
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            if(inner_best > best) {
                best = inner_best;
                x0_ret = x0_inner;
                y0_ret = y0_inner;
                x1_ret = x1_inner;
                y1_ret = y1_inner;
            }
        }
    }
    double px_x = (y1_ret - y0_ret) * (x1_ret - x0_ret) ;
    double px_y = allpx - px_x;
    px_x = 1 / px_x;
    px_y = 1 / px_y;
    double4_t sum_color_X, sum_color_Y;
    sum_color_X = prec_d[x1_ret + (nx + 1) * y1_ret] 
                - prec_d[x0_ret + (nx + 1) * y1_ret]
                - prec_d[x1_ret + (nx + 1) * y0_ret]
                + prec_d[x0_ret + (nx + 1) * y0_ret];
    sum_color_Y = colors_vec - sum_color_X;
    sum_color_Y *= px_y;
    sum_color_X *= px_x;
    Result result = {
        y0_ret, x0_ret, y1_ret, x1_ret, 
        {(float)sum_color_Y[0], (float)sum_color_Y[1], (float)sum_color_Y[2]}, 
        {(float)sum_color_X[0], (float)sum_color_X[1], (float)sum_color_X[2]}
    };

    free(prec_d);
    free(vecd_d);

    return result;
}
