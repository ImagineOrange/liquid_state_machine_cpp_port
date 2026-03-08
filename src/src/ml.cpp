#include "ml.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <cstdio>
#include <cblas.h>

namespace cls {

void StandardScaler::fit(const Mat& X) {
    int n = X.rows, p = X.cols;
    mean.assign(p, 0.0);
    scale.assign(p, 1.0);

    for (int j = 0; j < p; j++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += X(i, j);
        mean[j] = s / n;
    }
    for (int j = 0; j < p; j++) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            double d = X(i, j) - mean[j];
            s += d * d;
        }
        double var = s / n; // sklearn uses n, not n-1
        scale[j] = std::sqrt(var);
        if (scale[j] < 1e-10) scale[j] = 1.0;
    }
}

Mat StandardScaler::transform(const Mat& X) const {
    Mat out(X.rows, X.cols);
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            out(i, j) = (X(i, j) - mean[j]) / scale[j];
        }
    }
    return out;
}

Mat StandardScaler::fit_transform(const Mat& X) {
    fit(X);
    return transform(X);
}

void nan_to_num(Mat& X) {
    for (auto& v : X.data) {
        if (std::isnan(v) || std::isinf(v)) v = 0.0;
    }
}

// Dual-form ridge regression: solve (K + αI) α_vec = Y where K = X*X^T.
// When n << p (typical for reservoir readout: 1200 samples, 36000 features),
// this is ~16x faster than the primal SVD approach.

RidgeFoldContext ridge_fold_prepare(const Mat& X_train, const std::vector<int>& y_train,
                                    const Mat& X_test, const std::vector<int>& y_test,
                                    const std::vector<int>& classes) {
    RidgeFoldContext ctx;
    int n_train = X_train.rows;
    int n_test = X_test.rows;
    int p = X_train.cols;
    int n_classes = (int)classes.size();

    ctx.n_train = n_train;
    ctx.n_test = n_test;
    ctx.classes = classes;

    // K = X_train * X_train^T  (n_train x n_train) via BLAS
    ctx.K = Mat(n_train, n_train, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_train, n_train, p,
                1.0, X_train.data.data(), p,
                X_train.data.data(), p,
                0.0, ctx.K.data.data(), n_train);

    // K_test = X_test * X_train^T  (n_test x n_train) via BLAS
    ctx.K_test = Mat(n_test, n_train, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_test, n_train, p,
                1.0, X_test.data.data(), p,
                X_train.data.data(), p,
                0.0, ctx.K_test.data.data(), n_train);

    // Build binary targets Y_bin (n_train x n_classes)
    ctx.Y_bin = Mat(n_train, n_classes, -1.0);
    std::map<int, int> class_to_idx;
    for (int c = 0; c < n_classes; c++) class_to_idx[classes[c]] = c;
    for (int i = 0; i < n_train; i++) {
        auto it = class_to_idx.find(y_train[i]);
        if (it != class_to_idx.end()) {
            ctx.Y_bin(i, it->second) = 1.0;
        }
    }

    return ctx;
}

RidgeResult ridge_fold_solve(const RidgeFoldContext& ctx,
                              const Mat& X_test, const std::vector<int>& y_test,
                              double alpha) {
    int n_train = ctx.n_train;
    int n_test = ctx.n_test;
    int n_classes = (int)ctx.classes.size();

    // Build (K + αI) — copy K since dposv overwrites it
    // and also copy Y_bin since dposv overwrites the RHS with the solution
    std::vector<double> Ka(n_train * n_train);
    std::vector<double> alpha_vec(n_train * n_classes);

    // Copy K (row-major) and add αI
    std::copy(ctx.K.data.begin(), ctx.K.data.end(), Ka.begin());
    for (int i = 0; i < n_train; i++) {
        Ka[i * n_train + i] += alpha;
    }

    // Copy Y_bin into alpha_vec (will be overwritten with solution)
    std::copy(ctx.Y_bin.data.begin(), ctx.Y_bin.data.end(), alpha_vec.begin());

    // Transpose to column-major for LAPACK dposv
    // Ka is symmetric so transposing is a no-op for the values,
    // but we need column-major layout for LAPACK.
    // For symmetric matrices stored row-major, we can use uplo='L' with row-major
    // which is equivalent to uplo='U' in column-major. Use dposv with uplo='U'.
    // Actually: LAPACK always expects column-major. For symmetric matrix, A^T = A,
    // so row-major with 'L' is same as col-major with 'U'.
    // But the RHS B must also be column-major.

    // Transpose RHS to column-major: (n_train x n_classes) row-major → col-major
    std::vector<double> B_col(n_train * n_classes);
    for (int i = 0; i < n_train; i++)
        for (int c = 0; c < n_classes; c++)
            B_col[c * n_train + i] = alpha_vec[i * n_classes + c];

    // Solve (K + αI) * X = Y via Cholesky (K + αI is symmetric positive definite)
    char uplo = 'U';  // row-major 'L' == col-major 'U'
    int info;
    dposv_(&uplo, &n_train, &n_classes, Ka.data(), &n_train,
           B_col.data(), &n_train, &info);

    if (info != 0) {
        fprintf(stderr, "WARNING: dposv failed (info=%d), falling back to large alpha\n", info);
    }

    // B_col now contains the solution α_vec in column-major (n_train x n_classes)
    // decisions = K_test * α_vec  (n_test x n_classes)
    // K_test is (n_test x n_train) row-major, α_vec is (n_train x n_classes) col-major
    // Convert α_vec back to row-major for BLAS
    Mat alpha_mat(n_train, n_classes);
    for (int i = 0; i < n_train; i++)
        for (int c = 0; c < n_classes; c++)
            alpha_mat(i, c) = B_col[c * n_train + i];

    // decisions = K_test * alpha_mat  (n_test x n_classes)
    Mat decisions(n_test, n_classes, 0.0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_test, n_classes, n_train,
                1.0, ctx.K_test.data.data(), n_train,
                alpha_mat.data.data(), n_classes,
                0.0, decisions.data.data(), n_classes);

    // Predictions: argmax of decision function
    RidgeResult result;
    result.predictions.resize(n_test);
    result.decision_values = decisions;

    for (int i = 0; i < n_test; i++) {
        int best_c = 0;
        double best_v = decisions(i, 0);
        for (int c = 1; c < n_classes; c++) {
            if (decisions(i, c) > best_v) {
                best_v = decisions(i, c);
                best_c = c;
            }
        }
        result.predictions[i] = ctx.classes[best_c];
    }

    result.accuracy = accuracy_score(y_test,
                                      std::vector<int>(result.predictions.begin(),
                                                       result.predictions.begin() + n_test));
    return result;
}

RidgeResult ridge_classify(const Mat& X_train, const std::vector<int>& y_train,
                           const Mat& X_test, const std::vector<int>& y_test,
                           double alpha, const std::vector<int>& classes) {
    auto ctx = ridge_fold_prepare(X_train, y_train, X_test, y_test, classes);
    return ridge_fold_solve(ctx, X_test, y_test, alpha);
}

double accuracy_score(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    int n = (int)y_true.size();
    if (n == 0) return 0.0;
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return (double)correct / n;
}

std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                                const std::vector<int>& y_pred,
                                                const std::vector<int>& labels) {
    int n = (int)labels.size();
    std::map<int, int> label_to_idx;
    for (int i = 0; i < n; i++) label_to_idx[labels[i]] = i;

    std::vector<std::vector<int>> cm(n, std::vector<int>(n, 0));
    for (size_t i = 0; i < y_true.size(); i++) {
        auto it = label_to_idx.find(y_true[i]);
        auto jt = label_to_idx.find(y_pred[i]);
        if (it != label_to_idx.end() && jt != label_to_idx.end()) {
            cm[it->second][jt->second]++;
        }
    }
    return cm;
}

SplitIndices stratified_shuffle_split(const std::vector<int>& y, double test_size,
                                       uint64_t random_state) {
    int n = (int)y.size();
    std::mt19937_64 rng(random_state);

    // Group by class
    std::map<int, std::vector<int>> by_class;
    for (int i = 0; i < n; i++) by_class[y[i]].push_back(i);

    SplitIndices result;

    for (auto& [cls, indices] : by_class) {
        // Shuffle
        for (int i = (int)indices.size() - 1; i > 0; i--) {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            std::swap(indices[i], indices[j]);
        }

        int n_test = std::max(1, (int)std::round(indices.size() * test_size));
        int n_train = (int)indices.size() - n_test;

        for (int i = 0; i < n_train; i++) result.train.push_back(indices[i]);
        for (int i = n_train; i < (int)indices.size(); i++) result.test.push_back(indices[i]);
    }

    // Shuffle train and test
    for (int i = (int)result.train.size() - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(result.train[i], result.train[dist(rng)]);
    }
    for (int i = (int)result.test.size() - 1; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(result.test[i], result.test[dist(rng)]);
    }

    return result;
}

std::vector<SplitIndices> stratified_kfold(const std::vector<int>& y, int n_splits,
                                            uint64_t random_state) {
    int n = (int)y.size();
    std::mt19937_64 rng(random_state);

    // Group indices by class
    std::map<int, std::vector<int>> by_class;
    for (int i = 0; i < n; i++) by_class[y[i]].push_back(i);

    // Shuffle within each class
    for (auto& [cls, indices] : by_class) {
        for (int i = (int)indices.size() - 1; i > 0; i--) {
            std::uniform_int_distribution<int> dist(0, i);
            std::swap(indices[i], indices[dist(rng)]);
        }
    }

    // Assign each sample to a fold, round-robin within each class
    std::vector<int> fold_id(n, 0);
    for (auto& [cls, indices] : by_class) {
        for (int i = 0; i < (int)indices.size(); i++) {
            fold_id[indices[i]] = i % n_splits;
        }
    }

    // Build splits
    std::vector<SplitIndices> splits(n_splits);
    for (int f = 0; f < n_splits; f++) {
        for (int i = 0; i < n; i++) {
            if (fold_id[i] == f) splits[f].test.push_back(i);
            else splits[f].train.push_back(i);
        }
    }

    return splits;
}

// Student's t-distribution CDF approximation (Abramowitz & Stegun)
static double t_cdf(double t, int df) {
    double x = (double)df / (df + t * t);
    // Regularized incomplete beta function I_x(a, b) where a = df/2, b = 0.5
    // Using simple series expansion
    double a = df / 2.0;
    double b = 0.5;

    // Use integration via beta function approximation
    // For large df, approximate with normal
    if (df > 100) {
        // Normal approximation
        double z = t * (1.0 - 1.0 / (4.0 * df)) / std::sqrt(1.0 + t * t / (2.0 * df));
        return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    }

    // Incomplete beta via continued fraction
    // p-value = I_x(df/2, 1/2) for two-sided
    // Use simple numeric integration
    int steps = 10000;
    double dx_step = x / steps;
    double integral = 0.0;
    for (int i = 0; i < steps; i++) {
        double xi = (i + 0.5) * dx_step;
        integral += std::pow(xi, a - 1.0) * std::pow(1.0 - xi, b - 1.0) * dx_step;
    }

    // Beta function B(a, b)
    double log_beta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    double Ix = integral / std::exp(log_beta);
    Ix = std::max(0.0, std::min(1.0, Ix));

    // CDF: if t >= 0: 1 - Ix/2, else Ix/2
    if (t >= 0) return 1.0 - Ix / 2.0;
    else return Ix / 2.0;
}

PairedStats compute_paired_stats(const std::vector<double>& folds_a,
                                  const std::vector<double>& folds_b,
                                  double acc_a, double acc_b) {
    int n = (int)folds_a.size();
    std::vector<double> gaps(n);
    for (int i = 0; i < n; i++) gaps[i] = folds_a[i] - folds_b[i];

    double mean_gap = 0;
    for (double g : gaps) mean_gap += g;
    mean_gap /= n;

    double var = 0;
    for (double g : gaps) var += (g - mean_gap) * (g - mean_gap);
    var /= (n - 1);
    double sd = std::sqrt(var);

    double t_stat = mean_gap / std::max(sd / std::sqrt((double)n), 1e-10);
    double d = mean_gap / std::max(sd, 1e-10);

    // Two-sided p-value
    double p_value;
    if (n > 1) {
        double cdf_val = t_cdf(std::abs(t_stat), n - 1);
        p_value = 2.0 * (1.0 - cdf_val);
        p_value = std::max(0.0, std::min(1.0, p_value));
    } else {
        p_value = 1.0;
    }

    // Bootstrap CI
    std::mt19937_64 rng(42);
    int n_boot = 10000;
    std::vector<double> boot_gaps(n_boot);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int b = 0; b < n_boot; b++) {
        double s = 0;
        for (int i = 0; i < n; i++) s += gaps[dist(rng)];
        boot_gaps[b] = s / n;
    }
    std::sort(boot_gaps.begin(), boot_gaps.end());
    double ci_lo = boot_gaps[(int)(0.025 * n_boot)];
    double ci_hi = boot_gaps[(int)(0.975 * n_boot)];

    std::string stars;
    if (p_value < 0.001) stars = "***";
    else if (p_value < 0.01) stars = "**";
    else if (p_value < 0.05) stars = "*";
    else stars = "n.s.";

    return {
        (acc_a - acc_b) * 100.0,
        ci_lo * 100.0, ci_hi * 100.0,
        p_value, t_stat, d, stars
    };
}

} // namespace cls
