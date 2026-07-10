#include <gtest/gtest.h>

#include <dataset.hpp>
#include <knn.hpp>
#include <sstream>
#include <visualizer.hpp>

class VisualizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        data_ = dataset::generateSimpleDataset(20, 42);
        classifier_.setTrainingData(data_);
    }

    std::vector<Point> data_;
    Knn classifier_;
};

TEST_F(VisualizerTest, GetColorForLabel) {
    for (int i = 0; i < 10; ++i) {
        EXPECT_FALSE(
            visualizer::getColorForLabel(i, false).empty());
        EXPECT_FALSE(
            visualizer::getColorForLabel(i, true).empty());
    }
}

TEST_F(VisualizerTest, GetCharForLabel) {
    std::set<char> chars;
    for (int i = 0; i < 10; ++i)
        chars.insert(visualizer::getCharForLabel(i));
    EXPECT_GT(chars.size(), 1);
}

TEST_F(VisualizerTest, ProgressBar) {
    EXPECT_NE(visualizer::progressBar(0.0, 20)
                  .find("0.0%"),
              std::string::npos);
    EXPECT_NE(visualizer::progressBar(0.5, 20)
                  .find("50.0%"),
              std::string::npos);
    EXPECT_NE(visualizer::progressBar(1.0, 20)
                  .find("100.0%"),
              std::string::npos);
}

namespace {

void redirectCout(
    std::ostringstream& oss, std::streambuf*& old) {
    old = std::cout.rdbuf(oss.rdbuf());
}

void restoreCout(std::streambuf* old) {
    std::cout.rdbuf(old);
}

struct CoutGuard {
    std::ostringstream oss;
    std::streambuf* old;

    CoutGuard() { old = std::cout.rdbuf(oss.rdbuf()); }
    ~CoutGuard() { std::cout.rdbuf(old); }
};

}  // namespace

TEST_F(VisualizerTest, DrawMapNoCrash) {
    CoutGuard g;
    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;
    EXPECT_NO_THROW(visualizer::drawMap(classifier_, config));
}

TEST_F(VisualizerTest, DrawMapWithDataNoCrash) {
    CoutGuard g;
    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;
    EXPECT_NO_THROW(
        visualizer::drawMapWithData(classifier_, config));
}

TEST_F(VisualizerTest, DrawMapWithQueryNoCrash) {
    CoutGuard g;
    Point query(10.0, 10.0);
    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;
    EXPECT_NO_THROW(visualizer::drawMapWithQuery(
        classifier_, query, config));
}

TEST_F(VisualizerTest, DrawMapWithNeighborsNoCrash) {
    CoutGuard g;
    Point query(10.0, 10.0);
    VisualizerConfig config;
    config.gridSize = 10;
    config.k = 5;
    config.useColors = false;
    EXPECT_NO_THROW(visualizer::drawMapWithNeighbors(
        classifier_, query, config));
}

TEST_F(VisualizerTest, DrawDataPointsNoCrash) {
    CoutGuard g;
    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;
    EXPECT_NO_THROW(
        visualizer::drawDataPoints(data_, config));
}

TEST_F(VisualizerTest, EmptyClassifier) {
    CoutGuard g;
    Knn empty;
    VisualizerConfig config;
    config.gridSize = 10;
    EXPECT_NO_THROW(visualizer::drawMap(empty, config));
}

TEST_F(VisualizerTest, EmptyData) {
    CoutGuard g;
    std::vector<Point> empty;
    VisualizerConfig config;
    config.gridSize = 10;
    EXPECT_NO_THROW(
        visualizer::drawDataPoints(empty, config));
}

TEST_F(VisualizerTest, DrawSeparator) {
    CoutGuard g;
    EXPECT_NO_THROW(visualizer::drawSeparator(50, false));
    EXPECT_NO_THROW(visualizer::drawSeparator(50, true));
}

TEST_F(VisualizerTest, DrawLegend) {
    CoutGuard g;
    EXPECT_NO_THROW(visualizer::drawLegend(2, true));
    EXPECT_NO_THROW(visualizer::drawLegend(5, false));
}

TEST_F(VisualizerTest, DifferentGridSizes) {
    CoutGuard g;
    std::vector<int> sizes = {5, 10, 20, 40};

    for (int size : sizes) {
        VisualizerConfig config;
        config.gridSize = size;
        config.useColors = false;
        EXPECT_NO_THROW(
            visualizer::drawMap(classifier_, config));
    }
}

TEST_F(VisualizerTest, ConfigurationOptions) {
    CoutGuard g;
    VisualizerConfig config;
    config.gridSize = 15;
    config.k = 3;
    config.useColors = true;
    config.showLegend = true;
    config.showAxes = false;
    config.showGrid = false;
    config.emptyChar = '.';
    config.padding = 0.2;
    EXPECT_NO_THROW(
        visualizer::drawMap(classifier_, config));
}
