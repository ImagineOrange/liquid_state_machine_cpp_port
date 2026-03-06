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
}
#endif

namespace cls {

// Thread-local RNG
inline thread_local std::mt19937_64 g_rng;

inline void rng_seed(uint64_t seed) { g_rng.seed(seed); }

inline double rng_uniform() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(g_rng);
}

inline double rng_uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(g_rng);
}

inline double rng_normal(double mean = 0.0, double stddev = 1.0) {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(g_rng);
}

inline std::vector<double> rng_normal_vec(int n, double mean = 0.0, double stddev = 1.0) {
    std::normal_distribution<double> dist(mean, stddev);
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) out[i] = dist(g_rng);
    return out;
}

inline double rng_lognormal(double mu_log, double sigma_log) {
    std::lognormal_distribution<double> dist(mu_log, sigma_log);
    return dist(g_rng);
}

inline std::vector<int> rng_choice(int n, int k, bool replace = false) {
    std::vector<int> result;
    if (replace) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < k; i++) result.push_back(dist(g_rng));
    } else {
        std::vector<int> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        for (int i = 0; i < k; i++) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            int j = dist(g_rng);
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
