#include <gtest/gtest.h>

#include <dataset.hpp>
#include <point.hpp>

class DatasetTest : public ::testing::Test {
};

TEST_F(DatasetTest, GenerateSimpleDataset) {
    auto data = dataset::generateSimpleDataset(50, 42);

    EXPECT_EQ(data.size(), 100);

    int class_0 = 0, class_1 = 0;
    for (const auto& point : data) {
        if (point.label == 0) class_0++;
        else if (point.label == 1) class_1++;
    }

    EXPECT_EQ(class_0, 50);
    EXPECT_EQ(class_1, 50);
}

TEST_F(DatasetTest, GenerateClusters) {
    std::vector<ClusterConfig> configs = {
        ClusterConfig(0.0, 0.0, 1.0, 0, 30),
        ClusterConfig(10.0, 10.0, 1.0, 1, 40),
        ClusterConfig(5.0, 15.0, 0.5, 2, 20)};

    auto data = dataset::generateClusters(configs, 123);

    EXPECT_EQ(data.size(), 90);

    std::map<int, int> counts;
    for (const auto& p : data) counts[p.label]++;

    EXPECT_EQ(counts[0], 30);
    EXPECT_EQ(counts[1], 40);
    EXPECT_EQ(counts[2], 20);
}

TEST_F(DatasetTest, ReproducibilityWithSeed) {
    auto data_1 = dataset::generateSimpleDataset(100, 42);
    auto data_2 = dataset::generateSimpleDataset(100, 42);

    ASSERT_EQ(data_1.size(), data_2.size());

    for (size_t i = 0; i < data_1.size(); ++i) {
        EXPECT_DOUBLE_EQ(data_1[i].x, data_2[i].x);
        EXPECT_DOUBLE_EQ(data_1[i].y, data_2[i].y);
        EXPECT_EQ(data_1[i].label, data_2[i].label);
    }
}

TEST_F(DatasetTest, TrainTestSplit) {
    auto data = dataset::generateSimpleDataset(100, 42);
    auto [train, test]
        = dataset::trainTestSplit(data, 0.7, false);

    EXPECT_EQ(train.size(), 140);
    EXPECT_EQ(test.size(), 60);
    EXPECT_EQ(train.size() + test.size(), data.size());
}

TEST_F(DatasetTest, TrainTestSplitWithShuffle) {
    auto data = dataset::generateSimpleDataset(50, 42);
    auto [t1, s1]
        = dataset::trainTestSplit(data, 0.8, true, 123);
    auto [t2, s2]
        = dataset::trainTestSplit(data, 0.8, true, 456);

    EXPECT_EQ(t1.size(), t2.size());
    EXPECT_EQ(s1.size(), s2.size());

    bool different = false;
    size_t n = std::min(t1.size(), size_t(10));
    for (size_t i = 0; i < n; ++i)
        if (t1[i].x != t2[i].x || t1[i].y != t2[i].y) {
            different = true;
            break;
        }
    EXPECT_TRUE(different);
}

TEST_F(DatasetTest, CalculateAccuracy) {
    std::vector<int> predictions
        = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0};
    std::vector<int> ground_truth
        = {0, 1, 0, 0, 1, 1, 0, 1, 1, 0};

    double accuracy = dataset::calculateAccuracy(
        predictions, ground_truth);
    EXPECT_DOUBLE_EQ(accuracy, 80.0);
}

TEST_F(DatasetTest, PerfectAccuracy) {
    std::vector<int> predictions = {0, 1, 1, 0, 1};
    std::vector<int> gt = {0, 1, 1, 0, 1};
    EXPECT_DOUBLE_EQ(
        dataset::calculateAccuracy(predictions, gt), 100.0);
}

TEST_F(DatasetTest, ZeroAccuracy) {
    std::vector<int> predictions = {0, 0, 0, 0, 0};
    std::vector<int> gt = {1, 1, 1, 1, 1};
    EXPECT_DOUBLE_EQ(
        dataset::calculateAccuracy(predictions, gt), 0.0);
}

TEST_F(DatasetTest, GetDatasetStats) {
    auto data = dataset::generateSimpleDataset(50, 42);
    std::string stats = dataset::getDatasetStats(data);

    EXPECT_FALSE(stats.empty());
    EXPECT_NE(stats.find("Total points: 100"),
              std::string::npos);
    EXPECT_NE(stats.find("Number of classes: 2"),
              std::string::npos);
}

TEST_F(DatasetTest, ConfusionMatrix) {
    std::vector<int> predictions
        = {0, 0, 1, 1, 0, 1, 0, 1};
    std::vector<int> gt = {0, 1, 1, 0, 0, 1, 1, 1};

    auto matrix = dataset::confusionMatrix(
        predictions, gt, 2);

    EXPECT_EQ(matrix.size(), 2);
    EXPECT_EQ(matrix[0].size(), 2);
    EXPECT_EQ(matrix[0][0], 2);
    EXPECT_EQ(matrix[0][1], 1);
    EXPECT_EQ(matrix[1][0], 2);
    EXPECT_EQ(matrix[1][1], 3);
}

TEST_F(DatasetTest, Normalize) {
    std::vector<Point> data
        = {Point(0.0, 0.0, 0), Point(10.0, 20.0, 1),
           Point(5.0, 10.0, 0)};

    dataset::normalize(data);

    EXPECT_DOUBLE_EQ(data[0].x, 0.0);
    EXPECT_DOUBLE_EQ(data[0].y, 0.0);
    EXPECT_DOUBLE_EQ(data[1].x, 1.0);
    EXPECT_DOUBLE_EQ(data[1].y, 1.0);
}

TEST_F(DatasetTest, GetBounds) {
    std::vector<Point> data
        = {Point(-5.0, 3.0), Point(10.0, -2.0),
           Point(0.0, 15.0)};

    auto [min_x, max_x, min_y, max_y]
        = dataset::getBounds(data);

    EXPECT_DOUBLE_EQ(min_x, -5.0);
    EXPECT_DOUBLE_EQ(max_x, 10.0);
    EXPECT_DOUBLE_EQ(min_y, -2.0);
    EXPECT_DOUBLE_EQ(max_y, 15.0);
}

TEST_F(DatasetTest, SaveAndLoadCSV) {
    std::vector<Point> original
        = {Point(1.5, 2.5, 0), Point(3.7, 4.2, 1),
           Point(5.1, 6.9, 0)};

    std::string filename = "test_dataset.csv";

    EXPECT_TRUE(dataset::saveToCSV(original, filename));

    auto loaded = dataset::loadFromCSV(filename);

    ASSERT_EQ(loaded.size(), original.size());

    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_NEAR(loaded[i].x, original[i].x, 1e-5);
        EXPECT_NEAR(loaded[i].y, original[i].y, 1e-5);
        EXPECT_EQ(loaded[i].label, original[i].label);
    }

    std::remove(filename.c_str());
}
