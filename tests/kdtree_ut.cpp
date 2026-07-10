#include <gtest/gtest.h>
#include <kdtree.hpp>
#include <dataset.hpp>

class KdTreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_ = dataset::generateSimpleDataset(50, 42);
        tree_.build(data_);
    }

    std::vector<Point> data_;
    KdTree tree_;
};

TEST_F(KdTreeTest, BuildTree) {
    EXPECT_EQ(tree_.size(), 100);
    EXPECT_FALSE(tree_.empty());
}

TEST_F(KdTreeTest, KNearestNeighbors) {
    Point query(10.0, 10.0);
    auto neighbors = tree_.kNearestNeighbors(query, 5);

    EXPECT_EQ(neighbors.size(), 5);

    for (size_t i = 1; i < neighbors.size(); ++i)
        EXPECT_GE(neighbors[i].first, neighbors[i - 1].first);
}

TEST_F(KdTreeTest, RangeQuery) {
    auto results = tree_.rangeQuery(4.0, 6.0, 4.0, 6.0);

    for (const auto& point : results) {
        EXPECT_GE(point.x, 4.0);
        EXPECT_LE(point.x, 6.0);
        EXPECT_GE(point.y, 4.0);
        EXPECT_LE(point.y, 6.0);
    }
}

TEST_F(KdTreeTest, EmptyTree) {
    KdTree empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);
    EXPECT_TRUE(
        empty.kNearestNeighbors(Point(0, 0), 5).empty());
}
