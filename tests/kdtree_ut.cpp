#include <gtest/gtest.h>
#include <kdtree.hpp>
#include <dataset.hpp>

class KDTreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        data = Dataset::generateSimpleDataset(50, 42);
        tree.build(data);
    }

    std::vector<Point> data;
    KDTree tree;
};

TEST_F(KDTreeTest, BuildTree) {
    EXPECT_EQ(tree.size(), 100);
    EXPECT_FALSE(tree.empty());
}

TEST_F(KDTreeTest, KNearestNeighbors) {
    Point query(10.0, 10.0);
    auto neighbors = tree.kNearestNeighbors(query, 5);

    EXPECT_EQ(neighbors.size(), 5);

    // Should be sorted by distance
    for (size_t i = 1; i < neighbors.size(); ++i) {
        EXPECT_GE(neighbors[i].first, neighbors[i - 1].first);
    }
}

TEST_F(KDTreeTest, RangeQuery) {
    auto results = tree.rangeQuery(4.0, 6.0, 4.0, 6.0);

    // All results should be in range
    for (const auto& point : results) {
        EXPECT_GE(point.x, 4.0);
        EXPECT_LE(point.x, 6.0);
        EXPECT_GE(point.y, 4.0);
        EXPECT_LE(point.y, 6.0);
    }
}

TEST_F(KDTreeTest, EmptyTree) {
    KDTree emptyTree;
    EXPECT_TRUE(emptyTree.empty());
    EXPECT_EQ(emptyTree.size(), 0);

    Point query(0, 0);
    auto neighbors = emptyTree.kNearestNeighbors(query, 5);
    EXPECT_TRUE(neighbors.empty());
}
