#include <gtest/gtest.h>

#include <dataset.hpp>
#include <knn.hpp>
#include <point.hpp>

class KnnTest : public ::testing::Test {
protected:
    void SetUp() override {
        training_data_ = {Point(0.0, 0.0, 0), Point(1.0, 1.0, 0),
                          Point(2.0, 2.0, 0), Point(10.0, 10.0, 1),
                          Point(11.0, 11.0, 1), Point(12.0, 12.0, 1)};
        classifier_.setTrainingData(training_data_);
    }

    std::vector<Point> training_data_;
    Knn classifier_;
};

TEST_F(KnnTest, BasicPrediction) {
    Point q1(0.5, 0.5, -1);
    EXPECT_EQ(classifier_.predict(q1, 3), 0);

    Point q2(10.5, 10.5, -1);
    EXPECT_EQ(classifier_.predict(q2, 3), 1);
}

TEST_F(KnnTest, KEqualsOne) {
    Point query(0.1, 0.1, -1);
    EXPECT_EQ(classifier_.predict(query, 1), 0);
}

TEST_F(KnnTest, KEqualsDatasetSize) {
    Point query(5.0, 5.0, -1);
    int pred = classifier_.predict(query, training_data_.size());
    EXPECT_TRUE(pred == 0 || pred == 1);
}

TEST_F(KnnTest, EmptyTrainingData) {
    Knn empty;
    EXPECT_THROW(empty.predict(Point(1.0, 1.0), 3),
                 std::runtime_error);
}

TEST_F(KnnTest, InvalidK) {
    Point query(1.0, 1.0);
    EXPECT_THROW(classifier_.predict(query, 0),
                 std::invalid_argument);
    EXPECT_THROW(classifier_.predict(query, -1),
                 std::invalid_argument);
    EXPECT_THROW(classifier_.predict(query, 100),
                 std::invalid_argument);
}

TEST_F(KnnTest, GetKNearest) {
    Point query(0.0, 0.0);
    auto neighbors = classifier_.getKNearest(query, 3);

    EXPECT_EQ(neighbors.size(), 3);
    EXPECT_DOUBLE_EQ(neighbors[0].distance, 0.0);
    EXPECT_EQ(neighbors[0].label, 0);

    for (size_t i = 1; i < neighbors.size(); ++i)
        EXPECT_GE(neighbors[i].distance,
                  neighbors[i - 1].distance);
}

TEST_F(KnnTest, PredictWithConfidence) {
    Point query(0.5, 0.5);
    auto [label, confidence]
        = classifier_.predictWithConfidence(query, 3);
    EXPECT_EQ(label, 0);
    EXPECT_GT(confidence, 0.0);
    EXPECT_LE(confidence, 1.0);
}

TEST_F(KnnTest, BatchPrediction) {
    std::vector<Point> queries
        = {Point(0.5, 0.5), Point(10.5, 10.5),
           Point(1.5, 1.5)};
    auto predictions = classifier_.predictBatch(queries, 3);
    EXPECT_EQ(predictions.size(), 3);
    EXPECT_EQ(predictions[0], 0);
    EXPECT_EQ(predictions[1], 1);
    EXPECT_EQ(predictions[2], 0);
}

TEST_F(KnnTest, AccuracyOnSyntheticDataset) {
    auto data = dataset::generateSimpleDataset(100, 42);
    auto [train, test]
        = dataset::trainTestSplit(data, 0.7, true, 42);

    Knn knn(train);
    std::vector<int> predictions;
    std::vector<int> ground_truth;

    for (const auto& point : test) {
        predictions.push_back(knn.predict(point, 5));
        ground_truth.push_back(point.label);
    }

    double accuracy = dataset::calculateAccuracy(
        predictions, ground_truth);
    EXPECT_GT(accuracy, 90.0);
}

TEST_F(KnnTest, TieBreaking) {
    std::vector<Point> tie_data
        = {Point(0.0, 0.0, 0), Point(0.0, 1.0, 1)};
    Knn tie_clf(tie_data);

    Point query(0.0, 0.5);
    int pred = tie_clf.predict(query, 2);
    EXPECT_EQ(pred, 0);
}

TEST_F(KnnTest, HelperMethods) {
    EXPECT_TRUE(classifier_.isReady());
    EXPECT_EQ(classifier_.size(), 6);

    Knn empty;
    EXPECT_FALSE(empty.isReady());
    EXPECT_EQ(empty.size(), 0);
}

TEST_F(KnnTest, DifferentKValues) {
    Point query(5.0, 5.0);
    for (int k = 1;
         k <= static_cast<int>(training_data_.size()); ++k)
        EXPECT_NO_THROW(classifier_.predict(query, k));
}

TEST_F(KnnTest, ConfidenceRange) {
    std::vector<Point> queries
        = {Point(0.0, 0.0), Point(5.0, 5.0),
           Point(10.0, 10.0)};

    for (const auto& query : queries)
        for (int k = 1; k <= 5; ++k) {
            auto [label, confidence]
                = classifier_.predictWithConfidence(query, k);
            EXPECT_GE(confidence, 0.0);
            EXPECT_LE(confidence, 1.0);
        }
}
