#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#endif

#include <dataset.hpp>
#include <knn.hpp>
#include <knn_optimized.hpp>
#include <benchmark.hpp>
#include <point.hpp>
#include <visualizer.hpp>

using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using hires = std::chrono::high_resolution_clock;

static int64_t timeMs(auto duration) {
    return std::chrono::duration_cast<ms>(duration).count();
}

static int64_t timeUs(auto duration) {
    return std::chrono::duration_cast<us>(duration).count();
}

void testPointDistance() {
    std::cout << "Running Point distance tests...\n";

    Point p1(0.0, 0.0, 0);
    Point p2(3.0, 4.0, 1);
    double dist1 = p1.distanceTo(p2);
    assert(std::abs(dist1 - 5.0) < 1e-9
           && "Test 1 failed: 3-4-5 triangle");
    std::cout << "  \xE2\x9C\x93 Test 1: "
              << "Distance between " << p1
              << " and " << p2 << " = " << dist1 << "\n";

    Point p3(5.0, 7.0, 0);
    Point p4(5.0, 7.0, 1);
    double dist2 = p3.distanceTo(p4);
    assert(std::abs(dist2 - 0.0) < 1e-9
           && "Test 2 failed: same point");
    std::cout << "  \xE2\x9C\x93 Test 2: Distance between "
              << "same points = " << dist2 << "\n";

    Point p5(0.0, 5.0, 0);
    Point p6(10.0, 5.0, 0);
    double dist3 = p5.distanceTo(p6);
    assert(std::abs(dist3 - 10.0) < 1e-9
           && "Test 3 failed: horizontal distance");
    std::cout << "  \xE2\x9C\x93 Test 3: Horizontal distance = "
              << dist3 << "\n";

    std::cout << "All Point distance tests passed!"
              << " \xE2\x9C\x93\xE2\x9C\x93\xE2\x9C\x93\n\n";
}

void demoDatasetGeneration() {
    std::cout << "=== Dataset Generation Demo ===\n\n";

    std::cout << "Generating simple 2-class dataset...\n";
    auto dataset = dataset::generateSimpleDataset(150, 42);
    std::cout << dataset::getDatasetStats(dataset) << "\n\n";

    std::cout << "Generating custom 3-class dataset...\n";
    std::vector<ClusterConfig> configs = {
        ClusterConfig(5.0, 5.0, 1.2, 0, 100),
        ClusterConfig(15.0, 15.0, 1.5, 1, 100),
        ClusterConfig(10.0, 5.0, 1.0, 2, 80)};
    auto multiClassDataset
        = dataset::generateClusters(configs, 123);
    std::cout << dataset::getDatasetStats(multiClassDataset)
              << "\n\n";
}

void demoKNNClassifier() {
    std::cout << "=== Knn Classifier Demo ===\n\n";

    std::cout << "Generating training dataset...\n";
    auto fullDataset = dataset::generateSimpleDataset(200, 42);

    std::cout << "Splitting into train/test sets (80/20)...\n";
    auto [trainData, testData]
        = dataset::trainTestSplit(fullDataset, 0.8, true, 42);
    std::cout << "  Training set: " << trainData.size()
              << " points\n";
    std::cout << "  Test set: " << testData.size() << " points\n";
    std::cout << "\n";

    std::cout << "Training Knn classifier...\n";
    Knn classifier(trainData);
    std::cout << "  Classifier ready with " << classifier.size()
              << " training points\n\n";

    std::cout << "Testing different k values:\n";
    std::vector<int> kValues = {1, 3, 5, 7, 10};

    for (int k : kValues) {
        auto predictions = classifier.predictBatch(testData, k);
        std::vector<int> groundTruth;
        for (const auto& point : testData)
            groundTruth.push_back(point.label);

        double accuracy
            = dataset::calculateAccuracy(predictions, groundTruth);
        std::cout << "  k=" << std::setw(2) << k
                  << ": Accuracy = " << std::fixed
                  << std::setprecision(2) << accuracy << "%\n";
    }
    std::cout << "\n";

    std::cout << "Single prediction examples:\n";
    std::vector<Point> queryPoints
        = {Point(5.0, 5.0), Point(15.0, 15.0),
           Point(10.0, 10.0)};

    for (const auto& query : queryPoints) {
        auto [label, confidence]
            = classifier.predictWithConfidence(query, 5);
        std::cout << "  Query " << query << " -> Class " << label
                  << " (confidence: " << std::fixed
                  << std::setprecision(1) << (confidence * 100)
                  << "%)\n";
    }
    std::cout << "\n";

    std::cout << "K-Nearest neighbors for Point(10.0, 10.0):\n";
    Point query(10.0, 10.0);
    auto neighbors = classifier.getKNearest(query, 5);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << neighbors[i].point
                  << " - distance: " << std::fixed
                  << std::setprecision(3)
                  << neighbors[i].distance << "\n";
    }
    std::cout << "\n";
}

void benchmarkKNN() {
    std::cout << "=== Performance Benchmark ===\n\n";

    std::vector<int> sizes = {100, 500, 1000, 2000};

    for (int size : sizes) {
        auto data = dataset::generateSimpleDataset(size / 2, 42);
        auto [train, test]
            = dataset::trainTestSplit(data, 0.8, true, 42);

        Knn knn(train);

        auto start = hires::now();
        for (const auto& point : test)
            knn.predict(point, 5);
        auto end = hires::now();
        double avgMs = timeUs(end - start) / 1000.0 / test.size();

        std::cout << "  Dataset size: " << std::setw(4)
                  << train.size() << " | Test size: "
                  << std::setw(3) << test.size()
                  << " | Avg time per prediction: " << std::fixed
                  << std::setprecision(3) << avgMs << " ms\n";
    }
    std::cout << "\n";
}

void demoDatasetUtilities() {
    std::cout << "=== Dataset Utilities Demo ===\n\n";

    auto data = dataset::generateSimpleDataset(100, 42);

    std::cout << "Testing CSV save/load...\n";
    std::string filename = "demo_dataset.csv";

    if (dataset::saveToCSV(data, filename)) {
        std::cout << "  \xE2\x9C\x93 Saved data to " << filename
                  << "\n";

        auto loaded = dataset::loadFromCSV(filename);
        std::cout << "  \xE2\x9C\x93 Loaded " << loaded.size()
                  << " points from " << filename << "\n";

        std::remove(filename.c_str());
    }
    std::cout << "\n";

    std::cout << "Testing normalization...\n";
    auto [minXBefore, maxXBefore, minYBefore, maxYBefore]
        = dataset::getBounds(data);
    std::cout << "  Before: X[" << minXBefore << ", " << maxXBefore
              << "], Y[" << minYBefore << ", " << maxYBefore
              << "]\n";

    dataset::normalize(data);
    auto [minXAfter, maxXAfter, minYAfter, maxYAfter]
        = dataset::getBounds(data);
    std::cout << "  After:  X[" << minXAfter << ", " << maxXAfter
              << "], Y[" << minYAfter << ", " << maxYAfter
              << "]\n\n";

    std::cout << "Testing confusion matrix...\n";
    std::vector<int> predictions
        = {0, 0, 1, 1, 0, 1, 1, 0};
    std::vector<int> groundTruth
        = {0, 1, 1, 0, 0, 1, 1, 1};

    auto matrix = dataset::confusionMatrix(
        predictions, groundTruth, 2);
    std::cout << "  Confusion Matrix (2x2):\n";
    std::cout << "         Pred 0  Pred 1\n";
    std::cout << "  True 0:   " << matrix[0][0] << "       "
              << matrix[0][1] << "\n";
    std::cout << "  True 1:   " << matrix[1][0] << "       "
              << matrix[1][1] << "\n\n";
}

void demoVisualization() {
    std::cout << "=== Visualization Demo ===\n\n";

    auto data = dataset::generateSimpleDataset(50, 42);
    Knn classifier(data);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;
    config.showLegend = true;

    std::cout << "\nPress Enter for decision boundaries...";
    std::cin.get();
    visualizer::drawMap(classifier, config);

    std::cout << "\nPress Enter for decision boundaries "
                 "with training data...";
    std::cin.get();
    visualizer::drawMapWithData(classifier, config);

    std::cout << "\nPress Enter for scatter plot...";
    std::cin.get();
    visualizer::drawDataPoints(data, config);

    std::cout << "\nPress Enter for query prediction...";
    std::cin.get();
    Point q1(10.0, 10.0);
    visualizer::drawMapWithQuery(classifier, q1, config);

    std::cout << "\nPress Enter for k-nearest neighbors...";
    std::cin.get();
    Point q2(7.0, 8.0);
    visualizer::drawMapWithNeighbors(classifier, q2, config);

    std::cout << "\n";
}

void demoMultiClassVisualization() {
    std::cout << "=== Multi-Class Visualization Demo ===\n\n";

    std::cout << "Generating 4-class dataset...\n";

    std::vector<ClusterConfig> configs = {
        ClusterConfig(3.0, 3.0, 0.8, 0, 40),
        ClusterConfig(12.0, 3.0, 0.8, 1, 40),
        ClusterConfig(3.0, 12.0, 0.8, 2, 40),
        ClusterConfig(12.0, 12.0, 0.8, 3, 40)};

    auto data = dataset::generateClusters(configs, 42);
    Knn classifier(data);

    VisualizerConfig config;
    config.gridSize = 50;
    config.k = 7;
    config.useColors = true;

    std::cout << "\nPress Enter for 4-class boundaries...";
    std::cin.get();
    visualizer::drawMapWithData(classifier, config);

    std::cout << "\n";
}

void demoOptimizations() {
    std::cout << "=== Optimization Demo ===\n\n";

    auto data = dataset::generateSimpleDataset(200, 42);
    auto [train, test]
        = dataset::trainTestSplit(data, 0.8, true, 42);

    std::cout << "Comparing Standard vs Optimized Knn...\n\n";

    std::cout << "1. Standard Knn (linear search)...\n";
    Knn standardKNN(train);
    auto start = hires::now();
    standardKNN.predictBatch(test, 5);
    auto end = hires::now();
    int64_t d1 = timeMs(end - start);
    std::cout << "   Time: " << d1 << " ms\n";

    std::cout << "\n2. Optimized Knn (KD-tree)...\n";
    KnnOptimized optimizedKNN(train, true);
    start = hires::now();
    optimizedKNN.predictBatch(test, 5);
    end = hires::now();
    int64_t d2 = timeMs(end - start);
    std::cout << "   Time: " << d2 << " ms\n";
    std::cout << "   Speedup: " << std::fixed
              << std::setprecision(1)
              << (static_cast<double>(d1) / d2) << "x\n";

    std::cout << "\n3. Parallel prediction (4 threads)...\n";
    start = hires::now();
    optimizedKNN.predictBatchParallel(test, 5, 4);
    end = hires::now();
    int64_t d3 = timeMs(end - start);
    std::cout << "   Time: " << d3 << " ms\n";
    std::cout << "   Speedup vs standard: " << std::fixed
              << std::setprecision(1)
              << (static_cast<double>(d1) / d3) << "x\n";

    std::cout << "\n4. Testing weight strategies...\n";

    std::vector<std::pair<std::string, WeightStrategy>>
        strategies = {
            {"Uniform", WeightStrategy::UNIFORM},
            {"Inverse Distance",
             WeightStrategy::INVERSE_DISTANCE},
            {"Gaussian", WeightStrategy::GAUSSIAN}};

    for (const auto& [name, strategy] : strategies) {
        optimizedKNN.setWeightStrategy(strategy);
        auto preds = optimizedKNN.predictBatch(test, 5);

        std::vector<int> groundTruth;
        for (const auto& p : test)
            groundTruth.push_back(p.label);

        double acc
            = dataset::calculateAccuracy(preds, groundTruth);
        std::cout << "   " << std::setw(20) << std::left << name
                  << ": " << std::fixed << std::setprecision(1)
                  << acc << "%\n";
    }

    std::cout << "\n";
}

void demoCrossValidation() {
    std::cout << "=== Cross-Validation Demo ===\n\n";

    auto data = dataset::generateSimpleDataset(150, 42);

    std::cout << "Performing 5-fold CV...\n";
    auto accuracies
        = cross_validation::kFoldCV(data, 5, 5, 42);

    std::cout << "\nFold Results:\n";
    for (size_t i = 0; i < accuracies.size(); ++i) {
        std::cout << "  Fold " << (i + 1) << ": " << std::fixed
                  << std::setprecision(2) << accuracies[i]
                  << "%\n";
    }

    double mean
        = std::accumulate(
              accuracies.begin(), accuracies.end(), 0.0)
          / accuracies.size();
    std::cout << "\nMean Accuracy: " << std::fixed
              << std::setprecision(2) << mean << "%\n";

    std::cout << "\nFinding optimal k...\n";
    auto [bestK, bestAcc]
        = cross_validation::findOptimalK(data, 15, 5);
    std::cout << "Best k: " << bestK
              << " with accuracy: " << std::fixed
              << std::setprecision(2) << bestAcc << "%\n\n";
}

void interactiveVisualization() {
    std::cout << "=== Interactive Visualization ===\n\n";

    auto data = dataset::generateSimpleDataset(80, 123);
    Knn classifier(data);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;

    auto [minX, maxX, minY, maxY]
        = dataset::getBounds(data);

    std::cout << "Bounds: X[" << minX << ", " << maxX
              << "], Y[" << minY << ", " << maxY << "]\n";
    std::cout << "\nEnter query points to visualize.\n";
    std::cout << "Type 'exit' to quit.\n\n";

    std::string input;
    while (true) {
        std::cout << "Enter X Y (or 'exit'): ";
        std::getline(std::cin, input);

        if (input == "exit" || input == "quit" || input == "q")
            break;

        std::istringstream iss(input);
        double x, y;

        if (iss >> x >> y) {
            Point query(x, y);
            visualizer::clearScreen();
            visualizer::drawMapWithNeighbors(
                classifier, query, config);
            std::cout << "\n";
        } else {
            std::cout << "Invalid input. "
                         "Enter two numbers (x y).\n";
        }
    }

    std::cout << "\nExiting interactive mode.\n";
}

void printBanner() {
    std::cout
        << "=============================="
           "========\n"
        << "  Knn Classifier + "
           "Console Visualization\n"
        << "  v2.0\n"
        << "=============================="
           "========\n\n";
}

int showMenu() {
    std::cout << "Select demo mode:\n";
    std::cout << "1. Full demo (all features)\n";
    std::cout << "2. Visualization demo\n";
    std::cout << "3. Optimization demo\n";
    std::cout << "4. Cross-validation demo\n";
    std::cout << "5. Benchmark suite\n";
    std::cout << "6. Interactive mode\n";
    std::cout << "7. Quick test\n";
    std::cout << "\nChoice (1-7): ";

    int choice;
    std::cin >> choice;
    std::cin.ignore();
    return choice;
}

void printFooter() {
    std::cout
        << "\n=============================="
           "========\n"
        << "  All demos completed!\n"
        << "=============================="
           "========\n";
}

void demoMode(int choice) {
    switch (choice) {
        case 1:
            testPointDistance();
            demoDatasetGeneration();
            demoKNNClassifier();
            demoOptimizations();
            demoCrossValidation();
            demoDatasetUtilities();
            demoVisualization();
            demoMultiClassVisualization();
            std::cout << "\nPress Enter for benchmarks...";
            std::cin.get();
            benchmark::runFullSuite(1000, 200);
            break;

        case 2:
            demoVisualization();
            demoMultiClassVisualization();
            break;

        case 3:
            demoOptimizations();
            break;

        case 4:
            demoCrossValidation();
            break;

        case 5:
            benchmark::runFullSuite(2000, 400);
            break;

        case 6:
            interactiveVisualization();
            break;

        case 7:
            testPointDistance();
            {
                auto data
                    = dataset::generateSimpleDataset(50, 42);
                KnnOptimized classifier(data, true);
                VisualizerConfig config;
                config.gridSize = 35;
                config.k = 5;
                visualizer::drawMapWithData(
                    classifier, config);
            }
            break;

        default:
            std::cout << "Invalid choice. "
                         "Running quick test...\n";
            testPointDistance();
            break;
    }
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
#endif

    printBanner();
    int choice = showMenu();
    demoMode(choice);
    printFooter();

    return 0;
}