#pragma once
#include "common.h"
#include <vector>

namespace cls {

// StandardScaler: fit on train, transform train and test
struct StandardScaler {
    std::vector<double> mean;
    std::vector<double> scale; // std, clamped to avoid div-by-zero

    void fit(const Mat& X);
    Mat transform(const Mat& X) const;
    Mat fit_transform(const Mat& X);
};

// Replace NaN/Inf with 0
void nan_to_num(Mat& X);

// Ridge classifier (solver='svd') with multiclass one-vs-rest
struct RidgeResult {
    double accuracy;
    std::vector<int> predictions;
    Mat decision_values; // (n_samples, n_classes) or (n_samples, 1)
};

RidgeResult ridge_classify(const Mat& X_train, const std::vector<int>& y_train,
                           const Mat& X_test, const std::vector<int>& y_test,
                           double alpha, const std::vector<int>& classes);

// Pre-computed per-fold context: dual form (K = X*X^T) for fast multi-alpha sweeps.
// When n << p (typical for reservoir readout), solving the n×n dual system is
// dramatically faster than the p×p SVD (~16x for 1200×36240 matrices).
struct RidgeFoldContext {
    Mat K;            // (n_train x n_train) = X_train * X_train^T (Gram matrix)
    Mat K_test;       // (n_test x n_train) = X_test * X_train^T
    Mat Y_bin;        // (n_train x n_classes) one-hot targets
    int n_train;
    int n_test;
    std::vector<int> classes;
};

RidgeFoldContext ridge_fold_prepare(const Mat& X_train, const std::vector<int>& y_train,
                                    const Mat& X_test, const std::vector<int>& y_test,
                                    const std::vector<int>& classes);

RidgeResult ridge_fold_solve(const RidgeFoldContext& ctx,
                              const Mat& X_test, const std::vector<int>& y_test,
                              double alpha);

// Accuracy score
double accuracy_score(const std::vector<int>& y_true, const std::vector<int>& y_pred);

// Confusion matrix
std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                                const std::vector<int>& y_pred,
                                                const std::vector<int>& labels);

// StratifiedShuffleSplit: single split
struct SplitIndices {
    std::vector<int> train;
    std::vector<int> test;
};

SplitIndices stratified_shuffle_split(const std::vector<int>& y, double test_size,
                                       uint64_t random_state);

// StratifiedKFold: returns k folds where each sample appears in test exactly once
std::vector<SplitIndices> stratified_kfold(const std::vector<int>& y, int n_splits,
                                            uint64_t random_state);

// Paired t-test
struct PairedStats {
    double gap_pp;
    double ci_lo_pp, ci_hi_pp;
    double p_value;
    double t_stat;
    double cohens_d;
    std::string stars;
};

PairedStats compute_paired_stats(const std::vector<double>& folds_a,
                                  const std::vector<double>& folds_b,
                                  double acc_a, double acc_b);

} // namespace cls
