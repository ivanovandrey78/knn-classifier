#include <gtest/gtest.h>

#include <dataset.hpp>
#include <knn.hpp>
#include <point.hpp>

class KNNTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple training data
        trainingData = {Point(0.0, 0.0, 0), Point(1.0, 1.0, 0), Point(2.0, 2.0, 0),
                        Point(10.0, 10.0, 1), Point(11.0, 11.0, 1), Point(12.0, 12.0, 1)};

        classifier.setTrainingData(trainingData);
    }

    std::vector<Point> trainingData;
    KNN classifier;
};

// Test basic prediction
TEST_F(KNNTest, BasicPrediction) {
    // Query point close to class 0
    Point query1(0.5, 0.5, -1);
    int pred1 = classifier.predict(query1, 3);
    EXPECT_EQ(pred1, 0);

    // Query point close to class 1
    Point query2(10.5, 10.5, -1);
    int pred2 = classifier.predict(query2, 3);
    EXPECT_EQ(pred2, 1);
}

// Test with k=1
TEST_F(KNNTest, KEqualsOne) {
    Point query(0.1, 0.1, -1);
    int pred = classifier.predict(query, 1);
    EXPECT_EQ(pred, 0);  // Nearest is (0,0) with label 0
}

// Test with k equal to dataset size
TEST_F(KNNTest, KEqualsDatasetSize) {
    Point query(5.0, 5.0, -1);
    int pred = classifier.predict(query, trainingData.size());
    // Should still work, might be either class depending on voting
    EXPECT_TRUE(pred == 0 || pred == 1);
}

// Test exception on empty training data
TEST_F(KNNTest, EmptyTrainingData) {
    KNN emptyClassifier;
    Point query(1.0, 1.0);

    EXPECT_THROW(emptyClassifier.predict(query, 3), std::runtime_error);
}

// Test exception on invalid k
TEST_F(KNNTest, InvalidK) {
    Point query(1.0, 1.0);

    EXPECT_THROW(classifier.predict(query, 0), std::invalid_argument);
    EXPECT_THROW(classifier.predict(query, -1), std::invalid_argument);
    EXPECT_THROW(classifier.predict(query, 100), std::invalid_argument);
}

// Test getKNearest
TEST_F(KNNTest, GetKNearest) {
    Point query(0.0, 0.0);
    auto neighbors = classifier.getKNearest(query, 3);

    EXPECT_EQ(neighbors.size(), 3);

    // First neighbor should be at (0,0) with distance 0
    EXPECT_DOUBLE_EQ(neighbors[0].distance, 0.0);
    EXPECT_EQ(neighbors[0].label, 0);

    // Neighbors should be sorted by distance
    for (size_t i = 1; i < neighbors.size(); ++i) {
        EXPECT_GE(neighbors[i].distance, neighbors[i - 1].distance);
    }
}

// Test predictWithConfidence
TEST_F(KNNTest, PredictWithConfidence) {
    Point query(0.5, 0.5);
    auto [label, confidence] = classifier.predictWithConfidence(query, 3);

    EXPECT_EQ(label, 0);
    EXPECT_GT(confidence, 0.0);
    EXPECT_LE(confidence, 1.0);
}

// Test batch prediction
TEST_F(KNNTest, BatchPrediction) {
    std::vector<Point> queries = {Point(0.5, 0.5), Point(10.5, 10.5), Point(1.5, 1.5)};

    auto predictions = classifier.predictBatch(queries, 3);

    EXPECT_EQ(predictions.size(), 3);
    EXPECT_EQ(predictions[0], 0);
    EXPECT_EQ(predictions[1], 1);
    EXPECT_EQ(predictions[2], 0);
}

// Test accuracy on synthetic dataset
TEST_F(KNNTest, AccuracyOnSyntheticDataset) {
    // Generate well-separated dataset
    auto fullDataset = Dataset::generateSimpleDataset(100, 42);
    auto [train, test] = Dataset::trainTestSplit(fullDataset, 0.7, true, 42);

    KNN knn(train);

    // Predict on test set
    std::vector<int> predictions;
    std::vector<int> groundTruth;

    for (const auto& point : test) {
        predictions.push_back(knn.predict(point, 5));
        groundTruth.push_back(point.label);
    }

    double accuracy = Dataset::calculateAccuracy(predictions, groundTruth);

    // Well-separated clusters should have high accuracy
    EXPECT_GT(accuracy, 90.0);  // At least 90% accuracy
}

// Test tie-breaking (chooses smaller label)
TEST_F(KNNTest, TieBreaking) {
    std::vector<Point> tieData = {Point(0.0, 0.0, 0), Point(0.0, 1.0, 1)};

    KNN tieClassifier(tieData);

    // Query equidistant from both points
    Point query(0.0, 0.5);
    int pred = tieClassifier.predict(query, 2);

    // Should choose label 0 (smaller) in case of tie
    EXPECT_EQ(pred, 0);
}

// Test isReady and size methods
TEST_F(KNNTest, HelperMethods) {
    EXPECT_TRUE(classifier.isReady());
    EXPECT_EQ(classifier.size(), 6);

    KNN emptyClassifier;
    EXPECT_FALSE(emptyClassifier.isReady());
    EXPECT_EQ(emptyClassifier.size(), 0);
}

TEST_F(KNNTest, DifferentKValues) {
    Point query(5.0, 5.0);

    for (int k = 1; k <= static_cast<int>(trainingData.size()); ++k) {
        EXPECT_NO_THROW(classifier.predict(query, k));
    }
}

// Test confidence is always in valid range
TEST_F(KNNTest, ConfidenceRange) {
    std::vector<Point> queries = {Point(0.0, 0.0), Point(5.0, 5.0), Point(10.0, 10.0)};

    for (const auto& query : queries) {
        for (int k = 1; k <= 5; ++k) {
            auto [label, confidence] = classifier.predictWithConfidence(query, k);
            EXPECT_GE(confidence, 0.0);
            EXPECT_LE(confidence, 1.0);
        }
    }
}