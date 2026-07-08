#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <dataset.hpp>
#include <knn.hpp>
#include <point.hpp>
#include <visualizer.hpp>

/**
 * @brief Simple test function for Point distance calculation
 */
void testPointDistance() {
    std::cout << "Running Point distance tests..." << std::endl;

    // Test 1: Basic 3-4-5 triangle
    Point p1(0.0, 0.0, 0);
    Point p2(3.0, 4.0, 1);
    double dist1 = p1.distanceTo(p2);
    assert(std::abs(dist1 - 5.0) < 1e-9 && "Test 1 failed: 3-4-5 triangle");
    std::cout << "  ✓ Test 1: Distance between " << p1 << " and " << p2 << " = " << dist1
              << std::endl;

    // Test 2: Same point
    Point p3(5.0, 7.0, 0);
    Point p4(5.0, 7.0, 1);
    double dist2 = p3.distanceTo(p4);
    assert(std::abs(dist2 - 0.0) < 1e-9 && "Test 2 failed: same point");
    std::cout << "  ✓ Test 2: Distance between same points = " << dist2 << std::endl;

    // Test 3: Horizontal distance
    Point p5(0.0, 5.0, 0);
    Point p6(10.0, 5.0, 0);
    double dist3 = p5.distanceTo(p6);
    assert(std::abs(dist3 - 10.0) < 1e-9 && "Test 3 failed: horizontal distance");
    std::cout << "  ✓ Test 3: Horizontal distance = " << dist3 << std::endl;

    std::cout << "All Point distance tests passed! ✓✓✓" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Demo of dataset generation
 */
void demoDatasetGeneration() {
    std::cout << "=== Dataset Generation Demo ===" << std::endl;
    std::cout << std::endl;

    // Generate simple 2-class dataset
    std::cout << "Generating simple 2-class dataset..." << std::endl;
    auto dataset = Dataset::generateSimpleDataset(150, 42);
    std::cout << Dataset::getDatasetStats(dataset) << std::endl;
    std::cout << std::endl;

    // Generate custom multi-class dataset
    std::cout << "Generating custom 3-class dataset..." << std::endl;
    std::vector<ClusterConfig> configs = {ClusterConfig(5.0, 5.0, 1.2, 0, 100),
                                          ClusterConfig(15.0, 15.0, 1.5, 1, 100),
                                          ClusterConfig(10.0, 5.0, 1.0, 2, 80)};
    auto multiClassDataset = Dataset::generateClusters(configs, 123);
    std::cout << Dataset::getDatasetStats(multiClassDataset) << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Demo of KNN classifier
 */
void demoKNNClassifier() {
    std::cout << "=== KNN Classifier Demo ===" << std::endl;
    std::cout << std::endl;

    // Generate dataset
    std::cout << "Generating training dataset..." << std::endl;
    auto fullDataset = Dataset::generateSimpleDataset(200, 42);

    // Split into train and test
    std::cout << "Splitting into train/test sets (80/20)..." << std::endl;
    auto [trainData, testData] = Dataset::trainTestSplit(fullDataset, 0.8, true, 42);
    std::cout << "  Training set: " << trainData.size() << " points" << std::endl;
    std::cout << "  Test set: " << testData.size() << " points" << std::endl;
    std::cout << std::endl;

    // Create and train KNN classifier
    std::cout << "Training KNN classifier..." << std::endl;
    KNN classifier(trainData);
    std::cout << "  Classifier ready with " << classifier.size() << " training points"
              << std::endl;
    std::cout << std::endl;

    // Test different k values
    std::cout << "Testing different k values:" << std::endl;
    std::vector<int> kValues = {1, 3, 5, 7, 10};

    for (int k : kValues) {
        auto predictions = classifier.predictBatch(testData, k);

        std::vector<int> groundTruth;
        for (const auto& point : testData) {
            groundTruth.push_back(point.label);
        }

        double accuracy = Dataset::calculateAccuracy(predictions, groundTruth);
        std::cout << "  k=" << std::setw(2) << k << ": Accuracy = " << std::fixed
                  << std::setprecision(2) << accuracy << "%" << std::endl;
    }
    std::cout << std::endl;

    // Demo single prediction with confidence
    std::cout << "Single prediction examples:" << std::endl;
    std::vector<Point> queryPoints = {Point(5.0, 5.0), Point(15.0, 15.0), Point(10.0, 10.0)};

    for (const auto& query : queryPoints) {
        auto [label, confidence] = classifier.predictWithConfidence(query, 5);
        std::cout << "  Query " << query << " -> Class " << label << " (confidence: "
                  << std::fixed << std::setprecision(1) << (confidence * 100) << "%)"
                  << std::endl;
    }
    std::cout << std::endl;

    // Show k-nearest neighbors
    std::cout << "K-Nearest neighbors for Point(10.0, 10.0):" << std::endl;
    Point query(10.0, 10.0);
    auto neighbors = classifier.getKNearest(query, 5);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << neighbors[i].point << " - distance: "
                  << std::fixed << std::setprecision(3) << neighbors[i].distance << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Performance benchmark
 */
void benchmarkKNN() {
    std::cout << "=== Performance Benchmark ===" << std::endl;
    std::cout << std::endl;

    std::vector<int> datasetSizes = {100, 500, 1000, 2000};

    for (int size : datasetSizes) {
        auto dataset = Dataset::generateSimpleDataset(size / 2, 42);
        auto [train, test] = Dataset::trainTestSplit(dataset, 0.8, true, 42);

        KNN classifier(train);

        // Measure prediction time
        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& point : test) {
            classifier.predict(point, 5);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        double avgTimeMs = duration / 1000.0 / test.size();

        std::cout << "  Dataset size: " << std::setw(4) << train.size() << " | Test size: "
                  << std::setw(3) << test.size() << " | Avg time per prediction: " << std::fixed
                  << std::setprecision(3) << avgTimeMs << " ms" << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Demo of dataset utilities
 */
void demoDatasetUtilities() {
    std::cout << "=== Dataset Utilities Demo ===" << std::endl;
    std::cout << std::endl;

    // Generate dataset
    auto dataset = Dataset::generateSimpleDataset(100, 42);

    // Test CSV save/load
    std::cout << "Testing CSV save/load..." << std::endl;
    std::string filename = "demo_dataset.csv";

    if (Dataset::saveToCSV(dataset, filename)) {
        std::cout << "  ✓ Saved dataset to " << filename << std::endl;

        auto loaded = Dataset::loadFromCSV(filename);
        std::cout << "  ✓ Loaded " << loaded.size() << " points from " << filename << std::endl;

        // Cleanup
        std::remove(filename.c_str());
    }
    std::cout << std::endl;

    // Test normalization
    std::cout << "Testing normalization..." << std::endl;
    auto [minXBefore, maxXBefore, minYBefore, maxYBefore] = Dataset::getBounds(dataset);
    std::cout << "  Before: X[" << minXBefore << ", " << maxXBefore << "], Y[" << minYBefore
              << ", " << maxYBefore << "]" << std::endl;

    Dataset::normalize(dataset);
    auto [minXAfter, maxXAfter, minYAfter, maxYAfter] = Dataset::getBounds(dataset);
    std::cout << "  After:  X[" << minXAfter << ", " << maxXAfter << "], Y[" << minYAfter << ", "
              << maxYAfter << "]" << std::endl;
    std::cout << std::endl;

    // Test confusion matrix
    std::cout << "Testing confusion matrix..." << std::endl;
    std::vector<int> predictions = {0, 0, 1, 1, 0, 1, 1, 0};
    std::vector<int> groundTruth = {0, 1, 1, 0, 0, 1, 1, 1};

    auto matrix = Dataset::confusionMatrix(predictions, groundTruth, 2);
    std::cout << "  Confusion Matrix (2x2):" << std::endl;
    std::cout << "         Pred 0  Pred 1" << std::endl;
    std::cout << "  True 0:   " << matrix[0][0] << "       " << matrix[0][1] << std::endl;
    std::cout << "  True 1:   " << matrix[1][0] << "       " << matrix[1][1] << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  KNN Classifier with Console Visualization    ║" << std::endl;
    std::cout << "║  Production-Ready Implementation              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Step 2: Test Point class
    testPointDistance();

    // Step 3: Dataset generation
    demoDatasetGeneration();

    // Step 4: KNN classifier
    demoKNNClassifier();

    // Additional demos
    demoDatasetUtilities();
    benchmarkKNN();

    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  All demos completed successfully! ✓✓✓       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;

    return 0;
}