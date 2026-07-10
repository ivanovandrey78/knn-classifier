#include <gtest/gtest.h>
#include <knn_optimized.hpp>
#include <dataset.hpp>

class KNNOptimizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = dataset::generateSimpleDataset(50, 42);
        auto [train, test] = dataset::trainTestSplit(data, 0.8, true, 42);
        trainData = train;
        testData = test;
    }

    std::vector<Point> data;
    std::vector<Point> trainData;
    std::vector<Point> testData;
};

TEST_F(KNNOptimizedTest, KDTreePrediction) {
    KNNOptimized classifier(trainData, true);

    EXPECT_TRUE(classifier.hasKDTree());

    Point query(5.0, 5.0);
    int prediction = classifier.predict(query, 5);

    EXPECT_TRUE(prediction == 0 || prediction == 1);
}

TEST_F(KNNOptimizedTest, WeightedPrediction) {
    KNNOptimized classifier(trainData, true);
    classifier.setWeightStrategy(WeightStrategy::INVERSE_DISTANCE);

    Point query(5.0, 5.0);
    int prediction = classifier.predict(query, 5);

    EXPECT_TRUE(prediction == 0 || prediction == 1);
}

TEST_F(KNNOptimizedTest, ParallelBatchPrediction) {
    KNNOptimized classifier(trainData, true);

    auto predictions = classifier.predictBatchParallel(testData, 5, 4);

    EXPECT_EQ(predictions.size(), testData.size());
}

TEST_F(KNNOptimizedTest, cross_validation) {
    auto accuracies = cross_validation::kFoldCV(data, 5, 5, 42);

    EXPECT_EQ(accuracies.size(), 5);

    for (double acc : accuracies) {
        EXPECT_GE(acc, 0.0);
        EXPECT_LE(acc, 100.0);
    }
}

TEST_F(KNNOptimizedTest, FindOptimalK) {
    auto [bestK, bestAcc] = cross_validation::findOptimalK(data, 10, 3);

    EXPECT_GE(bestK, 1);
    EXPECT_LE(bestK, 10);
    EXPECT_GE(bestAcc, 0.0);
    EXPECT_LE(bestAcc, 100.0);
}
