#include <x86intrin.h>
#include <cmath>
#include <vector>

inline void* aligned_malloc(std::size_t bytes) {
    void* ret = nullptr;
    if (posix_memalign(&ret, 32, bytes)) {
        return nullptr;
    }
    return ret;
}

typedef float float8_t __attribute__ ((vector_size (32)));
const float8_t float8_0 = {0,0,0,0,0,0,0,0};

inline float8_t* float8_alloc(std::size_t n) {
    return static_cast<float8_t*>(aligned_malloc(sizeof(float8_t) * n));
}
static inline float8_t max8(float8_t x, float8_t y){ return x > y ? x : y;}
struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

struct RectShape{ int width; int height; int size; };

static inline float hmax8(float8_t vv) {
    float v = 0.f;
    for(int i=0; i < 8; i++) v = std::max(vv[i], v);
    return v;
}

std::vector<float> calculateSums(int ny, int nx, int ny_p, int nx_p, const float* data) {
    std::vector<float> sdata(nx_p * ny_p, 0.f);
    for(int y=0; y < ny; y++) {
        for(int x=0; x < nx; x++) {
          sdata[(x + 1) + nx_p * (y + 1)] = data[3 * (x + nx * y)]
                                          + sdata[(x + 1) + nx_p * y]
                                          + sdata[x + nx_p * (y + 1)]
                                          - sdata[x + nx_p * y];
        }
    }
    return sdata;
}

RectShape findRectShape(int ny, int nx, int ny_p, int nx_p, const float* sdata){
    int n = nx * ny;
    float vpc = sdata[nx_p * ny_p - 1];
    float8_t vvpc = {vpc, vpc, vpc, vpc, vpc, vpc, vpc, vpc};

    // Global best
    float gH = 0.f;
    int g_w = 0, g_h = 0;

    #pragma omp parallel
    {
        // Thread best
        float tH = 0.f;
        float8_t vtH = float8_0;

        #pragma omp for nowait schedule(dynamic, 1)
        for(int h=1; h <= ny; h++){
            for(int w=1; w <= nx; w++){
                int x_sz = h * w;
                int y_sz = n - x_sz;

                float x_inv = 1.0f / (float) x_sz;
                float y_inv = y_sz == 0 ? 0 : 1.0f / (float) y_sz;
                for(int y0=0; y0 <= ny - h; y0++){
                    int y1 = y0 + h;
                    int x0i = nx_p - w;
                    int vbatch = x0i / 8;
                    for(int i=0; i < vbatch; i++){
                        int x0 = 8 * i;
                        int x1 = x0 + w;

                        float8_t s1 = _mm256_loadu_ps(sdata + y1 * nx_p + x1);
                        float8_t s2 = _mm256_loadu_ps(sdata + y1 * nx_p + x0);
                        float8_t s3 = _mm256_loadu_ps(sdata + y0 * nx_p + x1);
                        float8_t s4 = _mm256_loadu_ps(sdata + y0 * nx_p + x0);

                        float8_t vvXc = s1 - s2 - s3 + s4;
                        float8_t vvYc = vvpc - vvXc;

                        vtH = max8(vtH, vvXc * vvXc * x_inv + vvYc * vvYc * y_inv);
                    }
                    for(int x0=8 * vbatch; x0 < x0i; x0++){
                        int x1 = x0 + w;

                        float s1 = sdata[y1 * nx_p + x1];
                        float s2 = sdata[y1 * nx_p + x0];
                        float s3 = sdata[y0 * nx_p + x1];
                        float s4 = sdata[y0 * nx_p + x0];

                        float vXc = s1 - s2 - s3 + s4;
                        float vYc = vpc - vXc;
                        float H = vXc * vXc * x_inv 
                                + vYc * vYc * y_inv;
                        if(H > tH) tH = H;
                    }
                }
                float H = hmax8(vtH);
                if(H > tH) tH = H;
                #pragma omp critical
                {
                    if(tH > gH) { 
                        gH = tH; 
                        g_w = w; 
                        g_h = h; 
                    }
                }
            }
        }
    }
    RectShape shape = {g_w, g_h, g_w * g_h};
    return shape;
}

Result segment(int ny, int nx, const float* data){
    int nx_p = nx + 1, ny_p = ny + 1;

    std::vector<float> sdata = calculateSums(ny, nx, ny_p, nx_p, data);
    RectShape rect = findRectShape(ny, nx, ny_p, nx_p, sdata.data());

    // Rectangle
    int n = ny * nx;
    float vpc = sdata[nx_p * ny_p - 1];

    int h = rect.height;
    int w = rect.width;
    int r_sz = rect.size;
    float r_inv = 1.0f / (float) r_sz;

    // Background
    int b_sz = n - r_sz;
    float b_inv = b_sz == 0 ? 0 : 1.0f / (float) b_sz;

    float gH = 0.f, r_c = 0.f, b_c = 0.f;
    int x0_ret = 0, x1_ret = 0, y0_ret = 0, y1_ret = 0;

    for(int y0=0; y0 <= ny - h; y0++) {
        for(int x0 = 0; x0 <= nx - w; x0++) {
            int y1 = y0 + h;
            int x1 = x0 + w;

            float s1 = sdata[y1 * nx_p + x1];
            float s2 = sdata[y1 * nx_p + x0];
            float s3 = sdata[y0 * nx_p + x1];
            float s4 = sdata[y0 * nx_p + x0];

            float vrc = s1 - s2 - s3 + s4;
            float vbc = vpc - vrc;

            float H = vrc * vrc * r_inv 
                    + vbc * vbc * b_inv;

            if(H > gH){
                gH = H;
                r_c = vrc * r_inv;
                b_c = vbc * b_inv;
                x0_ret = x0;
                x1_ret = x1;
                y0_ret = y0;
                y1_ret = y1;
            }
        }
    }
    Result result = { y0_ret, x0_ret, y1_ret, x1_ret, {b_c, b_c, b_c}, {r_c, r_c, r_c} };
    return result;
}
