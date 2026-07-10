#include <gtest/gtest.h>

#include <dataset.hpp>
#include <knn.hpp>
#include <sstream>
#include <visualizer.hpp>

class VisualizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple dataset
        data = dataset::generateSimpleDataset(20, 42);
        classifier.setTrainingData(data);
    }

    std::vector<Point> data;
    KNN classifier;
};

// Test color generation for labels
TEST_F(VisualizerTest, GetColorForLabel) {
    // Should not crash and should return non-empty strings
    for (int i = 0; i < 10; ++i) {
        std::string color = visualizer::getColorForLabel(i, false);
        EXPECT_FALSE(color.empty());

        std::string bgColor = visualizer::getColorForLabel(i, true);
        EXPECT_FALSE(bgColor.empty());
    }
}

// Test character generation for labels
TEST_F(VisualizerTest, GetCharForLabel) {
    // Should return different characters for different labels
    std::set<char> chars;
    for (int i = 0; i < 10; ++i) {
        char c = visualizer::getCharForLabel(i);
        chars.insert(c);
    }

    // Should have multiple unique characters
    EXPECT_GT(chars.size(), 1);
}

// Test progress bar
TEST_F(VisualizerTest, ProgressBar) {
    std::string bar0 = visualizer::progressBar(0.0, 20);
    EXPECT_FALSE(bar0.empty());
    EXPECT_NE(bar0.find("0.0%"), std::string::npos);

    std::string bar50 = visualizer::progressBar(0.5, 20);
    EXPECT_FALSE(bar50.empty());
    EXPECT_NE(bar50.find("50.0%"), std::string::npos);

    std::string bar100 = visualizer::progressBar(1.0, 20);
    EXPECT_FALSE(bar100.empty());
    EXPECT_NE(bar100.find("100.0%"), std::string::npos);
}

// Test drawMap doesn't crash
TEST_F(VisualizerTest, DrawMapNoCrash) {
    // Redirect stdout to prevent console output in tests
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    VisualizerConfig config;
    config.gridSize = 10;  // Small for testing
    config.useColors = false;  // No colors in test

    EXPECT_NO_THROW(visualizer::drawMap(classifier, config));

    std::cout.rdbuf(old);
}

// Test drawMapWithData doesn't crash
TEST_F(VisualizerTest, DrawMapWithDataNoCrash) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;

    EXPECT_NO_THROW(visualizer::drawMapWithData(classifier, config));

    std::cout.rdbuf(old);
}

// Test drawMapWithQuery doesn't crash
TEST_F(VisualizerTest, DrawMapWithQueryNoCrash) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    Point query(10.0, 10.0);
    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;

    EXPECT_NO_THROW(visualizer::drawMapWithQuery(classifier, query, config));

    std::cout.rdbuf(old);
}

// Test drawMapWithNeighbors doesn't crash
TEST_F(VisualizerTest, DrawMapWithNeighborsNoCrash) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    Point query(10.0, 10.0);
    VisualizerConfig config;
    config.gridSize = 10;
    config.k = 5;
    config.useColors = false;

    EXPECT_NO_THROW(visualizer::drawMapWithNeighbors(classifier, query, config));

    std::cout.rdbuf(old);
}

// Test drawDataPoints doesn't crash
TEST_F(VisualizerTest, DrawDataPointsNoCrash) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    VisualizerConfig config;
    config.gridSize = 10;
    config.useColors = false;

    EXPECT_NO_THROW(visualizer::drawDataPoints(data, config));

    std::cout.rdbuf(old);
}

// Test with empty classifier
TEST_F(VisualizerTest, EmptyClassifier) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    KNN emptyClassifier;
    VisualizerConfig config;
    config.gridSize = 10;

    // Should handle empty classifier gracefully
    EXPECT_NO_THROW(visualizer::drawMap(emptyClassifier, config));

    std::cout.rdbuf(old);
}

// Test with empty data
TEST_F(VisualizerTest, EmptyData) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    std::vector<Point> emptyData;
    VisualizerConfig config;
    config.gridSize = 10;

    // Should handle empty data gracefully
    EXPECT_NO_THROW(visualizer::drawDataPoints(emptyData, config));

    std::cout.rdbuf(old);
}

// Test separator drawing
TEST_F(VisualizerTest, DrawSeparator) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    EXPECT_NO_THROW(visualizer::drawSeparator(50, false));
    EXPECT_NO_THROW(visualizer::drawSeparator(50, true));

    std::cout.rdbuf(old);
}

// Test legend drawing
TEST_F(VisualizerTest, DrawLegend) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    EXPECT_NO_THROW(visualizer::drawLegend(2, true));
    EXPECT_NO_THROW(visualizer::drawLegend(5, false));

    std::cout.rdbuf(old);
}

// Test with different grid sizes
TEST_F(VisualizerTest, DifferentGridSizes) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    std::vector<int> gridSizes = {5, 10, 20, 40};

    for (int size : gridSizes) {
        VisualizerConfig config;
        config.gridSize = size;
        config.useColors = false;

        EXPECT_NO_THROW(visualizer::drawMap(classifier, config));
    }

    std::cout.rdbuf(old);
}

// Test configuration options
TEST_F(VisualizerTest, ConfigurationOptions) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());

    VisualizerConfig config;
    config.gridSize = 15;
    config.k = 3;
    config.useColors = true;
    config.showLegend = true;
    config.showAxes = false;
    config.showGrid = false;
    config.emptyChar = '.';
    config.padding = 0.2;

    EXPECT_NO_THROW(visualizer::drawMap(classifier, config));

    std::cout.rdbuf(old);
}