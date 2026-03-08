#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline void omp_set_num_threads(int) {}
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" {
    void dgesdd_(const char* jobz, const int* m, const int* n, double* a,
                 const int* lda, double* s, double* u, const int* ldu,
                 double* vt, const int* ldvt, double* work, const int* lwork,
                 int* iwork, int* info);
    void dposv_(const char* uplo, const int* n, const int* nrhs, double* a,
                const int* lda, double* b, const int* ldb, int* info);
}
#endif

namespace cls {

// ============================================================
// Fast thread-local RNG: xoshiro256+ with cached Box-Muller
// ============================================================
struct FastRng {
    uint64_t s[4];
    double normal_spare;
    bool has_spare;

    void seed(uint64_t seed) {
        // SplitMix64 to initialize state from a single seed
        has_spare = false;
        for (int i = 0; i < 4; i++) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    inline uint64_t next() {
        const uint64_t result = s[0] + s[3];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 45) | (s[3] >> 19); // rotl
        return result;
    }

    inline double uniform() {
        return (next() >> 11) * 0x1.0p-53; // [0, 1)
    }

    inline double uniform(double lo, double hi) {
        return lo + uniform() * (hi - lo);
    }

    inline double normal() {
        if (has_spare) {
            has_spare = false;
            return normal_spare;
        }
        // Box-Muller: generates two normals, caches one
        double u, v, s_val;
        do {
            u = 2.0 * uniform() - 1.0;
            v = 2.0 * uniform() - 1.0;
            s_val = u * u + v * v;
        } while (s_val >= 1.0 || s_val == 0.0);
        double mul = std::sqrt(-2.0 * std::log(s_val) / s_val);
        normal_spare = v * mul;
        has_spare = true;
        return u * mul;
    }

    inline double normal(double mean, double stddev) {
        return mean + stddev * normal();
    }

    inline double lognormal(double mu_log, double sigma_log) {
        return std::exp(normal(mu_log, sigma_log));
    }
};

inline thread_local FastRng g_fast_rng;

// Also keep std::mt19937_64 for non-hot-path use (rng_choice, etc.)
inline thread_local std::mt19937_64 g_rng;

inline void rng_seed(uint64_t seed) {
    g_rng.seed(seed);
    g_fast_rng.seed(seed);
}

// Hot-path functions use FastRng
inline double rng_uniform() { return g_fast_rng.uniform(); }
inline double rng_uniform(double lo, double hi) { return g_fast_rng.uniform(lo, hi); }
inline double rng_normal(double mean = 0.0, double stddev = 1.0) {
    return (mean == 0.0 && stddev == 1.0) ? g_fast_rng.normal() : g_fast_rng.normal(mean, stddev);
}

inline std::vector<double> rng_normal_vec(int n, double mean = 0.0, double stddev = 1.0) {
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) out[i] = g_fast_rng.normal(mean, stddev);
    return out;
}

inline double rng_lognormal(double mu_log, double sigma_log) {
    return g_fast_rng.lognormal(mu_log, sigma_log);
}

inline std::vector<int> rng_choice(int n, int k, bool replace = false) {
    std::vector<int> result;
    if (replace) {
        for (int i = 0; i < k; i++)
            result.push_back((int)(g_fast_rng.uniform() * n));
    } else {
        std::vector<int> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        for (int i = 0; i < k; i++) {
            int j = i + (int)(g_fast_rng.uniform() * (n - i));
            std::swap(pool[i], pool[j]);
        }
        result.assign(pool.begin(), pool.begin() + k);
    }
    return result;
}

// Audio sample
struct AudioSample {
    std::vector<double> spike_times_ms;
    std::vector<int32_t> freq_bin_indices;
    int digit;
    std::string speaker;
    std::string filename;
};

// 2D matrix (row-major)
struct Mat {
    std::vector<double> data;
    int rows = 0, cols = 0;

    Mat() = default;
    Mat(int r, int c, double val = 0.0) : data(r * c, val), rows(r), cols(c) {}

    double& operator()(int r, int c) { return data[r * cols + c]; }
    double operator()(int r, int c) const { return data[r * cols + c]; }

    void resize(int r, int c, double val = 0.0) {
        rows = r; cols = c;
        data.assign(r * c, val);
    }

    void fill(double val) { std::fill(data.begin(), data.end(), val); }
};

// Utility: argsort
template<typename T>
inline std::vector<int> argsort(const std::vector<T>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return v[a] < v[b]; });
    return idx;
}

template<typename T>
inline std::vector<int> argsort(const T* v, int n) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return v[a] < v[b]; });
    return idx;
}

// Clip
inline double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

inline void clip_vec(std::vector<double>& v, double lo, double hi) {
    for (auto& x : v) x = clip(x, lo, hi);
}

// SVD wrapper using LAPACK dgesdd
// X is m x n (row-major), returns S (singular values), U (m x min(m,n)), Vt (min(m,n) x n)
inline void svd_econ(const Mat& X, std::vector<double>& S, Mat& U, Mat& Vt) {
    int m = X.rows, n = X.cols;
    int k = std::min(m, n);

    // LAPACK expects column-major, so transpose
    std::vector<double> A(m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[j * m + i] = X(i, j);

    S.resize(k);
    std::vector<double> u_data(m * k);
    std::vector<double> vt_data(k * n);
    std::vector<int> iwork(8 * k);
    int info;

    // Query optimal work size
    double work_query;
    int lwork = -1;
    char jobz = 'S';
    dgesdd_(&jobz, &m, &n, A.data(), &m, S.data(),
            u_data.data(), &m, vt_data.data(), &k,
            &work_query, &lwork, iwork.data(), &info);

    lwork = (int)work_query + 1;
    std::vector<double> work(lwork);
    dgesdd_(&jobz, &m, &n, A.data(), &m, S.data(),
            u_data.data(), &m, vt_data.data(), &k,
            work.data(), &lwork, iwork.data(), &info);

    if (info != 0) {
        fprintf(stderr, "WARNING: dgesdd failed with info=%d\n", info);
    }

    // Convert back to row-major
    U = Mat(m, k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U(i, j) = u_data[j * m + i];

    Vt = Mat(k, n);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            Vt(i, j) = vt_data[j * k + i];
}

// JSON writer helpers
inline void json_write_string(FILE* f, const std::string& s) {
    fputc('"', f);
    for (char c : s) {
        if (c == '"') fprintf(f, "\\\"");
        else if (c == '\\') fprintf(f, "\\\\");
        else if (c == '\n') fprintf(f, "\\n");
        else fputc(c, f);
    }
    fputc('"', f);
}

inline void json_write_double(FILE* f, double v) {
    if (std::isnan(v)) fprintf(f, "null");
    else if (std::isinf(v)) fprintf(f, "null");
    else fprintf(f, "%.10g", v);
}

inline void json_write_int(FILE* f, int64_t v) {
    fprintf(f, "%lld", (long long)v);
}

inline void json_write_double_array(FILE* f, const std::vector<double>& v) {
    fputc('[', f);
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        json_write_double(f, v[i]);
    }
    fputc(']', f);
}

inline void json_write_int_array(FILE* f, const std::vector<int>& v) {
    fputc('[', f);
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "%d", v[i]);
    }
    fputc(']', f);
}

// Timer
inline double now_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace cls
