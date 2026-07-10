#include <gtest/gtest.h>
#include <knn_optimized.hpp>
#include <dataset.hpp>

class KnnOptimizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_ = dataset::generateSimpleDataset(50, 42);
        auto [train, test]
            = dataset::trainTestSplit(data_, 0.8, true, 42);
        train_data_ = train;
        test_data_ = test;
    }

    std::vector<Point> data_;
    std::vector<Point> train_data_;
    std::vector<Point> test_data_;
};

TEST_F(KnnOptimizedTest, KdTreePrediction) {
    KnnOptimized clf(train_data_, true);
    EXPECT_TRUE(clf.has_kd_tree());

    Point query(5.0, 5.0);
    int pred = clf.predict(query, 5);
    EXPECT_TRUE(pred == 0 || pred == 1);
}

TEST_F(KnnOptimizedTest, WeightedPrediction) {
    KnnOptimized clf(train_data_, true);
    clf.setWeightStrategy(WeightStrategy::INVERSE_DISTANCE);

    Point query(5.0, 5.0);
    int pred = clf.predict(query, 5);
    EXPECT_TRUE(pred == 0 || pred == 1);
}

TEST_F(KnnOptimizedTest, ParallelBatchPrediction) {
    KnnOptimized clf(train_data_, true);
    auto predictions
        = clf.predictBatchParallel(test_data_, 5, 4);
    EXPECT_EQ(predictions.size(), test_data_.size());
}

TEST_F(KnnOptimizedTest, CrossValidation) {
    auto accuracies
        = cross_validation::kFoldCV(data_, 5, 5, 42);
    EXPECT_EQ(accuracies.size(), 5);

    for (double acc : accuracies) {
        EXPECT_GE(acc, 0.0);
        EXPECT_LE(acc, 100.0);
    }
}

TEST_F(KnnOptimizedTest, FindOptimalK) {
    auto [best_k, best_acc]
        = cross_validation::findOptimalK(data_, 10, 3);
    EXPECT_GE(best_k, 1);
    EXPECT_LE(best_k, 10);
    EXPECT_GE(best_acc, 0.0);
    EXPECT_LE(best_acc, 100.0);
}
