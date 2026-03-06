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
