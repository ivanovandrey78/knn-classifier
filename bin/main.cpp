#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <dataset.hpp>
#include <knn.hpp>
#include <knn_optimized.hpp>
#include <benchmark.hpp>
#include <point.hpp>
#include <visualizer.hpp>

void testPointDistance() {
    std::cout << "Running Point distance tests..." << std::endl;

    Point p1(0.0, 0.0, 0);
    Point p2(3.0, 4.0, 1);
    double dist1 = p1.distanceTo(p2);
    assert(std::abs(dist1 - 5.0) < 1e-9
           && "Test 1 failed: 3-4-5 triangle");
    std::cout << "  ✓ Test 1: Distance between " << p1 << " and "
              << p2 << " = " << dist1 << std::endl;

    Point p3(5.0, 7.0, 0);
    Point p4(5.0, 7.0, 1);
    double dist2 = p3.distanceTo(p4);
    assert(std::abs(dist2 - 0.0) < 1e-9
           && "Test 2 failed: same point");
    std::cout << "  ✓ Test 2: Distance between same points = "
              << dist2 << std::endl;

    Point p5(0.0, 5.0, 0);
    Point p6(10.0, 5.0, 0);
    double dist3 = p5.distanceTo(p6);
    assert(std::abs(dist3 - 10.0) < 1e-9
           && "Test 3 failed: horizontal distance");
    std::cout << "  ✓ Test 3: Horizontal distance = " << dist3
              << std::endl;

    std::cout << "All Point distance tests passed! ✓✓✓"
              << std::endl;
    std::cout << std::endl;
}

void demoDatasetGeneration() {
    std::cout << "=== Dataset Generation Demo ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Generating simple 2-class dataset..." << std::endl;
    auto dataset = dataset::generateSimpleDataset(150, 42);
    std::cout << dataset::getDatasetStats(dataset) << std::endl;
    std::cout << std::endl;

    std::cout << "Generating custom 3-class dataset..." << std::endl;
    std::vector<ClusterConfig> configs = {
        ClusterConfig(5.0, 5.0, 1.2, 0, 100),
        ClusterConfig(15.0, 15.0, 1.5, 1, 100),
        ClusterConfig(10.0, 5.0, 1.0, 2, 80)};
    auto multiClassDataset
        = dataset::generateClusters(configs, 123);
    std::cout << dataset::getDatasetStats(multiClassDataset)
              << std::endl;
    std::cout << std::endl;
}

void demoKNNClassifier() {
    std::cout << "=== KNN Classifier Demo ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Generating training dataset..." << std::endl;
    auto fullDataset = dataset::generateSimpleDataset(200, 42);

    std::cout << "Splitting into train/test sets (80/20)..."
              << std::endl;
    auto [trainData, testData]
        = dataset::trainTestSplit(fullDataset, 0.8, true, 42);
    std::cout << "  Training set: " << trainData.size()
              << " points" << std::endl;
    std::cout << "  Test set: " << testData.size() << " points"
              << std::endl;
    std::cout << std::endl;

    std::cout << "Training KNN classifier..." << std::endl;
    KNN classifier(trainData);
    std::cout << "  Classifier ready with " << classifier.size()
              << " training points" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing different k values:" << std::endl;
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
                  << std::setprecision(2) << accuracy << "%"
                  << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Single prediction examples:" << std::endl;
    std::vector<Point> queryPoints
        = {Point(5.0, 5.0), Point(15.0, 15.0), Point(10.0, 10.0)};

    for (const auto& query : queryPoints) {
        auto [label, confidence]
            = classifier.predictWithConfidence(query, 5);
        std::cout << "  Query " << query << " -> Class " << label
                  << " (confidence: " << std::fixed
                  << std::setprecision(1) << (confidence * 100)
                  << "%)" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "K-Nearest neighbors for Point(10.0, 10.0):"
              << std::endl;
    Point query(10.0, 10.0);
    auto neighbors = classifier.getKNearest(query, 5);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << neighbors[i].point
                  << " - distance: " << std::fixed
                  << std::setprecision(3)
                  << neighbors[i].distance << std::endl;
    }
    std::cout << std::endl;
}

void benchmarkKNN() {
    std::cout << "=== Performance Benchmark ===" << std::endl;
    std::cout << std::endl;

    std::vector<int> datasetSizes = {100, 500, 1000, 2000};

    for (int size : datasetSizes) {
        auto dataset = dataset::generateSimpleDataset(size / 2, 42);
        auto [train, test]
            = dataset::trainTestSplit(dataset, 0.8, true, 42);

        KNN classifier(train);

        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& point : test)
            classifier.predict(point, 5);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration
            = std::chrono::duration_cast<
                std::chrono::microseconds>(end - start).count();

        double avgTimeMs = duration / 1000.0 / test.size();

        std::cout << "  Dataset size: " << std::setw(4)
                  << train.size() << " | Test size: "
                  << std::setw(3) << test.size()
                  << " | Avg time per prediction: " << std::fixed
                  << std::setprecision(3) << avgTimeMs << " ms"
                  << std::endl;
    }
    std::cout << std::endl;
}

void demoDatasetUtilities() {
    std::cout << "=== Dataset Utilities Demo ===" << std::endl;
    std::cout << std::endl;

    auto dataset = dataset::generateSimpleDataset(100, 42);

    std::cout << "Testing CSV save/load..." << std::endl;
    std::string filename = "demo_dataset.csv";

    if (dataset::saveToCSV(dataset, filename)) {
        std::cout << "  ✓ Saved dataset to " << filename
                  << std::endl;

        auto loaded = dataset::loadFromCSV(filename);
        std::cout << "  ✓ Loaded " << loaded.size()
                  << " points from " << filename << std::endl;

        std::remove(filename.c_str());
    }
    std::cout << std::endl;

    std::cout << "Testing normalization..." << std::endl;
    auto [minXBefore, maxXBefore, minYBefore, maxYBefore]
        = dataset::getBounds(dataset);
    std::cout << "  Before: X[" << minXBefore << ", " << maxXBefore
              << "], Y[" << minYBefore << ", " << maxYBefore << "]"
              << std::endl;

    dataset::normalize(dataset);
    auto [minXAfter, maxXAfter, minYAfter, maxYAfter]
        = dataset::getBounds(dataset);
    std::cout << "  After:  X[" << minXAfter << ", " << maxXAfter
              << "], Y[" << minYAfter << ", " << maxYAfter << "]"
              << std::endl;
    std::cout << std::endl;

    std::cout << "Testing confusion matrix..." << std::endl;
    std::vector<int> predictions
        = {0, 0, 1, 1, 0, 1, 1, 0};
    std::vector<int> groundTruth
        = {0, 1, 1, 0, 0, 1, 1, 1};

    auto matrix = dataset::confusionMatrix(
        predictions, groundTruth, 2);
    std::cout << "  Confusion Matrix (2x2):" << std::endl;
    std::cout << "         Pred 0  Pred 1" << std::endl;
    std::cout << "  True 0:   " << matrix[0][0] << "       "
              << matrix[0][1] << std::endl;
    std::cout << "  True 1:   " << matrix[1][0] << "       "
              << matrix[1][1] << std::endl;
    std::cout << std::endl;
}

void demoVisualization() {
    std::cout << "=== Visualization Demo ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Generating dataset for visualization..."
              << std::endl;
    auto dataset = dataset::generateSimpleDataset(50, 42);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;
    config.showLegend = true;

    std::cout << "\nPress Enter to see decision boundary "
                 "visualization...";
    std::cin.get();
    visualizer::drawMap(classifier, config);

    std::cout << "\nPress Enter to see decision boundary "
                 "with training data...";
    std::cin.get();
    visualizer::drawMapWithData(classifier, config);

    std::cout << "\nPress Enter to see training data "
                 "scatter plot...";
    std::cin.get();
    visualizer::drawDataPoints(dataset, config);

    std::cout << "\nPress Enter to see query point "
                 "prediction...";
    std::cin.get();
    Point query1(10.0, 10.0);
    visualizer::drawMapWithQuery(classifier, query1, config);

    std::cout << "\nPress Enter to see k-nearest neighbors...";
    std::cin.get();
    Point query2(7.0, 8.0);
    visualizer::drawMapWithNeighbors(
        classifier, query2, config);

    std::cout << std::endl;
}

void demoMultiClassVisualization() {
    std::cout << "=== Multi-Class Visualization Demo ==="
              << std::endl;
    std::cout << std::endl;

    std::cout << "Generating 4-class dataset..." << std::endl;

    std::vector<ClusterConfig> configs = {
        ClusterConfig(3.0, 3.0, 0.8, 0, 40),
        ClusterConfig(12.0, 3.0, 0.8, 1, 40),
        ClusterConfig(3.0, 12.0, 0.8, 2, 40),
        ClusterConfig(12.0, 12.0, 0.8, 3, 40)};

    auto dataset = dataset::generateClusters(configs, 42);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 50;
    config.k = 7;
    config.useColors = true;

    std::cout << "\nPress Enter to see 4-class "
                 "decision boundaries...";
    std::cin.get();
    visualizer::drawMapWithData(classifier, config);

    std::cout << std::endl;
}

void demoOptimizations() {
    std::cout << "=== Optimization Demo ===" << std::endl;
    std::cout << std::endl;

    auto dataset = dataset::generateSimpleDataset(200, 42);
    auto [train, test]
        = dataset::trainTestSplit(dataset, 0.8, true, 42);

    std::cout << "Comparing Standard KNN vs Optimized KNN..."
              << std::endl;
    std::cout << std::endl;

    std::cout << "1. Standard KNN (linear search)..."
              << std::endl;
    KNN standardKNN(train);
    auto start = std::chrono::high_resolution_clock::now();
    auto predictions1 = standardKNN.predictBatch(test, 5);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1
        = std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count();
    std::cout << "   Time: " << duration1 << " ms" << std::endl;

    std::cout << "\n2. Optimized KNN (KD-tree)..." << std::endl;
    KNNOptimized optimizedKNN(train, true);
    start = std::chrono::high_resolution_clock::now();
    auto predictions2 = optimizedKNN.predictBatch(test, 5);
    end = std::chrono::high_resolution_clock::now();
    auto duration2
        = std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count();
    std::cout << "   Time: " << duration2 << " ms" << std::endl;
    std::cout << "   Speedup: " << std::fixed
              << std::setprecision(1)
              << (static_cast<double>(duration1) / duration2)
              << "x" << std::endl;

    std::cout << "\n3. Parallel prediction (4 threads)..."
              << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto predictions3
        = optimizedKNN.predictBatchParallel(test, 5, 4);
    end = std::chrono::high_resolution_clock::now();
    auto duration3
        = std::chrono::duration_cast<
            std::chrono::milliseconds>(end - start).count();
    std::cout << "   Time: " << duration3 << " ms" << std::endl;
    std::cout << "   Speedup vs standard: " << std::fixed
              << std::setprecision(1)
              << (static_cast<double>(duration1) / duration3)
              << "x" << std::endl;

    std::cout << "\n4. Testing weight strategies..."
              << std::endl;

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
                  << acc << "%" << std::endl;
    }

    std::cout << std::endl;
}

void demoCrossValidation() {
    std::cout << "=== Cross-Validation Demo ===" << std::endl;
    std::cout << std::endl;

    auto dataset = dataset::generateSimpleDataset(150, 42);

    std::cout << "Performing 5-fold cross-validation..."
              << std::endl;
    auto accuracies
        = cross_validation::kFoldCV(dataset, 5, 5, 42);

    std::cout << "\nFold Results:" << std::endl;
    for (size_t i = 0; i < accuracies.size(); ++i) {
        std::cout << "  Fold " << (i + 1) << ": " << std::fixed
                  << std::setprecision(2) << accuracies[i] << "%"
                  << std::endl;
    }

    double mean
        = std::accumulate(
              accuracies.begin(), accuracies.end(), 0.0)
          / accuracies.size();
    std::cout << "\nMean Accuracy: " << std::fixed
              << std::setprecision(2) << mean << "%" << std::endl;

    std::cout << "\nFinding optimal k..." << std::endl;
    auto [bestK, bestAcc]
        = cross_validation::findOptimalK(dataset, 15, 5);
    std::cout << "Best k: " << bestK
              << " with accuracy: " << std::fixed
              << std::setprecision(2) << bestAcc << "%"
              << std::endl;

    std::cout << std::endl;
}

void interactiveVisualization() {
    std::cout << "=== Interactive Visualization ===" << std::endl;
    std::cout << std::endl;

    auto dataset = dataset::generateSimpleDataset(80, 123);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;

    auto [minX, maxX, minY, maxY]
        = dataset::getBounds(dataset);

    std::cout << "Dataset bounds: X[" << minX << ", " << maxX
              << "], Y[" << minY << ", " << maxY << "]"
              << std::endl;
    std::cout << "\nEnter query points to visualize predictions."
              << std::endl;
    std::cout << "Type 'exit' to quit.\n" << std::endl;

    std::string input;
    while (true) {
        std::cout << "Enter X Y coordinates (or 'exit'): ";
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
            std::cout << "Invalid input. Please enter "
                         "two numbers (x y)." << std::endl;
        }
    }

    std::cout << "\nExiting interactive mode." << std::endl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  KNN Classifier + Console Visualization  ║"
              << std::endl;
    std::cout << "║  v2.0                                    ║"
              << std::endl;
    std::cout << "╚══════════════════════════════════════════╝"
              << std::endl;
    std::cout << std::endl;

    std::cout << "Select demo mode:" << std::endl;
    std::cout << "1. Full demo (all features)" << std::endl;
    std::cout << "2. Visualization demo" << std::endl;
    std::cout << "3. Optimization demo "
                 "(KD-tree, parallel, weighted)" << std::endl;
    std::cout << "4. Cross-validation demo" << std::endl;
    std::cout << "5. Benchmark suite" << std::endl;
    std::cout << "6. Interactive mode" << std::endl;
    std::cout << "7. Quick test" << std::endl;
    std::cout << "\nChoice (1-7): ";

    int choice;
    std::cin >> choice;
    std::cin.ignore();

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
            std::cout << "\nPress Enter to run benchmarks...";
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
                auto dataset
                    = dataset::generateSimpleDataset(50, 42);
                KNNOptimized classifier(dataset, true);
                VisualizerConfig config;
                config.gridSize = 35;
                config.k = 5;
                visualizer::drawMapWithData(
                    classifier, config);
            }
            break;

        default:
            std::cout << "Invalid choice. "
                         "Running quick test..." << std::endl;
            testPointDistance();
            break;
    }

    std::cout << "\n╔══════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║  All demos completed successfully! ✓✓✓   ║"
              << std::endl;
    std::cout << "╚══════════════════════════════════════════╝"
              << std::endl;

    return 0;
}
