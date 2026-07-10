#include <gtest/gtest.h>

#include <dataset.hpp>
#include <point.hpp>

class DatasetTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test simple dataset generation
TEST_F(DatasetTest, GenerateSimpleDataset) {
    auto data = dataset::generateSimpleDataset(50, 42);

    EXPECT_EQ(data.size(), 100);  // 50 per class * 2 classes

    // Count labels
    int class0 = 0, class1 = 0;
    for (const auto& point : data) {
        if (point.label == 0)
            class0++;
        else if (point.label == 1)
            class1++;
    }

    EXPECT_EQ(class0, 50);
    EXPECT_EQ(class1, 50);
}

// Test cluster generation with custom configs
TEST_F(DatasetTest, GenerateClusters) {
    std::vector<ClusterConfig> configs = {ClusterConfig(0.0, 0.0, 1.0, 0, 30),
                                          ClusterConfig(10.0, 10.0, 1.0, 1, 40),
                                          ClusterConfig(5.0, 15.0, 0.5, 2, 20)};

    auto data = dataset::generateClusters(configs, 123);

    EXPECT_EQ(data.size(), 90);  // 30 + 40 + 20

    // Count each class
    std::map<int, int> counts;
    for (const auto& p : data) {
        counts[p.label]++;
    }

    EXPECT_EQ(counts[0], 30);
    EXPECT_EQ(counts[1], 40);
    EXPECT_EQ(counts[2], 20);
}

// Test reproducibility with same seed
TEST_F(DatasetTest, ReproducibilityWithSeed) {
    auto data1 = dataset::generateSimpleDataset(100, 42);
    auto data2 = dataset::generateSimpleDataset(100, 42);

    ASSERT_EQ(data1.size(), data2.size());

    for (size_t i = 0; i < data1.size(); ++i) {
        EXPECT_DOUBLE_EQ(data1[i].x, data2[i].x);
        EXPECT_DOUBLE_EQ(data1[i].y, data2[i].y);
        EXPECT_EQ(data1[i].label, data2[i].label);
    }
}

// Test train-test split
TEST_F(DatasetTest, TrainTestSplit) {
    auto data = dataset::generateSimpleDataset(100, 42);
    auto [train, test] = dataset::trainTestSplit(data, 0.7, false);

    EXPECT_EQ(train.size(), 140);  // 70% of 200
    EXPECT_EQ(test.size(), 60);    // 30% of 200
    EXPECT_EQ(train.size() + test.size(), data.size());
}

// Test train-test split with shuffle
TEST_F(DatasetTest, TrainTestSplitWithShuffle) {
    auto data = dataset::generateSimpleDataset(50, 42);
    auto [train1, test1] = dataset::trainTestSplit(data, 0.8, true, 123);
    auto [train2, test2] = dataset::trainTestSplit(data, 0.8, true, 456);

    // Same sizes
    EXPECT_EQ(train1.size(), train2.size());
    EXPECT_EQ(test1.size(), test2.size());

    // Different order (with high probability)
    bool different = false;
    for (size_t i = 0; i < std::min(train1.size(), size_t(10)); ++i) {
        if (train1[i].x != train2[i].x || train1[i].y != train2[i].y) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// Test accuracy calculation
TEST_F(DatasetTest, CalculateAccuracy) {
    std::vector<int> predictions = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0};
    std::vector<int> groundTruth = {0, 1, 0, 0, 1, 1, 0, 1, 1, 0};

    double accuracy = dataset::calculateAccuracy(predictions, groundTruth);

    // 8 correct out of 10 = 80%
    EXPECT_DOUBLE_EQ(accuracy, 80.0);
}

// Test perfect accuracy
TEST_F(DatasetTest, PerfectAccuracy) {
    std::vector<int> predictions = {0, 1, 1, 0, 1};
    std::vector<int> groundTruth = {0, 1, 1, 0, 1};

    double accuracy = dataset::calculateAccuracy(predictions, groundTruth);
    EXPECT_DOUBLE_EQ(accuracy, 100.0);
}

// Test zero accuracy
TEST_F(DatasetTest, ZeroAccuracy) {
    std::vector<int> predictions = {0, 0, 0, 0, 0};
    std::vector<int> groundTruth = {1, 1, 1, 1, 1};

    double accuracy = dataset::calculateAccuracy(predictions, groundTruth);
    EXPECT_DOUBLE_EQ(accuracy, 0.0);
}

// Test dataset statistics
TEST_F(DatasetTest, GetDatasetStats) {
    auto data = dataset::generateSimpleDataset(50, 42);
    std::string stats = dataset::getDatasetStats(data);

    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Total points: 100"), std::string::npos);
    EXPECT_NE(stats.find("Number of classes: 2"), std::string::npos);
}

// Test confusion matrix
TEST_F(DatasetTest, ConfusionMatrix) {
    std::vector<int> predictions = {0, 0, 1, 1, 0, 1, 0, 1};
    std::vector<int> groundTruth = {0, 1, 1, 0, 0, 1, 1, 1};

    auto matrix = dataset::confusionMatrix(predictions, groundTruth, 2);

    EXPECT_EQ(matrix.size(), 2);
    EXPECT_EQ(matrix[0].size(), 2);

    // True class 0: predicted 0 (2 times), predicted 1 (1 time)
    EXPECT_EQ(matrix[0][0], 2);
    EXPECT_EQ(matrix[0][1], 1);

    // True class 1: predicted 0 (2 times), predicted 1 (3 times)
    EXPECT_EQ(matrix[1][0], 2);
    EXPECT_EQ(matrix[1][1], 3);
}

// Test normalization
TEST_F(DatasetTest, Normalize) {
    std::vector<Point> data = {Point(0.0, 0.0, 0), Point(10.0, 20.0, 1), Point(5.0, 10.0, 0)};

    dataset::normalize(data);

    // Check all values are in [0, 1]
    for (const auto& p : data) {
        EXPECT_GE(p.x, 0.0);
        EXPECT_LE(p.x, 1.0);
        EXPECT_GE(p.y, 0.0);
        EXPECT_LE(p.y, 1.0);
    }

    // Check specific values
    EXPECT_DOUBLE_EQ(data[0].x, 0.0);  // min x -> 0
    EXPECT_DOUBLE_EQ(data[0].y, 0.0);  // min y -> 0
    EXPECT_DOUBLE_EQ(data[1].x, 1.0);  // max x -> 1
    EXPECT_DOUBLE_EQ(data[1].y, 1.0);  // max y -> 1
}

// Test bounds calculation
TEST_F(DatasetTest, GetBounds) {
    std::vector<Point> data = {Point(-5.0, 3.0), Point(10.0, -2.0), Point(0.0, 15.0)};

    auto [minX, maxX, minY, maxY] = dataset::getBounds(data);

    EXPECT_DOUBLE_EQ(minX, -5.0);
    EXPECT_DOUBLE_EQ(maxX, 10.0);
    EXPECT_DOUBLE_EQ(minY, -2.0);
    EXPECT_DOUBLE_EQ(maxY, 15.0);
}

// Test CSV save and load
TEST_F(DatasetTest, SaveAndLoadCSV) {
    std::vector<Point> original = {Point(1.5, 2.5, 0), Point(3.7, 4.2, 1), Point(5.1, 6.9, 0)};

    std::string filename = "test_dataset.csv";

    // Save
    bool saved = dataset::saveToCSV(original, filename);
    EXPECT_TRUE(saved);

    // Load
    auto loaded = dataset::loadFromCSV(filename);

    ASSERT_EQ(loaded.size(), original.size());

    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(loaded[i].x, original[i].x, 1e-5);
        EXPECT_NEAR(loaded[i].y, original[i].y, 1e-5);
        EXPECT_EQ(loaded[i].label, original[i].label);
    }

    // Cleanup
    std::remove(filename.c_str());
}