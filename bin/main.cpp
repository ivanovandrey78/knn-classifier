#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <dataset.hpp>
#include <knn.hpp>
#include <point.hpp>
#include <visualizer.hpp>

#ifdef _WIN32
    #include <windows.h>
    #include <fcntl.h>
    #include <io.h>
#endif

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

    Point p3(5.0, 7.0, 0);
    Point p4(5.0, 7.0, 1);
    double dist2 = p3.distanceTo(p4);
    assert(std::abs(dist2 - 0.0) < 1e-9 && "Test 2 failed: same point");
    std::cout << "  ✓ Test 2: Distance between same points = " << dist2 << std::endl;

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

    std::cout << "Generating training dataset..." << std::endl;
    auto fullDataset = Dataset::generateSimpleDataset(200, 42);

    std::cout << "Splitting into train/test sets (80/20)..." << std::endl;
    auto [trainData, testData] = Dataset::trainTestSplit(fullDataset, 0.8, true, 42);
    std::cout << "  Training set: " << trainData.size() << " points" << std::endl;
    std::cout << "  Test set: " << testData.size() << " points" << std::endl;
    std::cout << std::endl;

    std::cout << "Training KNN classifier..." << std::endl;
    KNN classifier(trainData);
    std::cout << "  Classifier ready with " << classifier.size() << " training points"
              << std::endl;
    std::cout << std::endl;

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

    std::cout << "Single prediction examples:" << std::endl;
    std::vector<Point> queryPoints = {Point(5.0, 5.0), Point(15.0, 15.0), Point(10.0, 10.0)};

    for (const auto& query : queryPoints) {
        auto [label, confidence] = classifier.predictWithConfidence(query, 5);
        std::cout << "  Query " << query << " -> Class " << label << " (confidence: "
                  << std::fixed << std::setprecision(1) << (confidence * 100) << "%)"
                  << std::endl;
    }
    std::cout << std::endl;

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

    auto dataset = Dataset::generateSimpleDataset(100, 42);

    std::cout << "Testing CSV save/load..." << std::endl;
    std::string filename = "demo_dataset.csv";

    if (Dataset::saveToCSV(dataset, filename)) {
        std::cout << "  ✓ Saved dataset to " << filename << std::endl;

        auto loaded = Dataset::loadFromCSV(filename);
        std::cout << "  ✓ Loaded " << loaded.size() << " points from " << filename << std::endl;

        std::remove(filename.c_str());
    }
    std::cout << std::endl;

    std::cout << "Testing normalization..." << std::endl;
    auto [minXBefore, maxXBefore, minYBefore, maxYBefore] = Dataset::getBounds(dataset);
    std::cout << "  Before: X[" << minXBefore << ", " << maxXBefore << "], Y[" << minYBefore
              << ", " << maxYBefore << "]" << std::endl;

    Dataset::normalize(dataset);
    auto [minXAfter, maxXAfter, minYAfter, maxYAfter] = Dataset::getBounds(dataset);
    std::cout << "  After:  X[" << minXAfter << ", " << maxXAfter << "], Y[" << minYAfter << ", "
              << maxYAfter << "]" << std::endl;
    std::cout << std::endl;

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

/**
 * @brief Demo of visualization capabilities
 */
void demoVisualization() {
    std::cout << "=== Visualization Demo ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Generating dataset for visualization..." << std::endl;
    auto dataset = Dataset::generateSimpleDataset(50, 42);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;
    config.showLegend = true;

    std::cout << "\nPress Enter to see decision boundary visualization...";
    std::cin.get();

    Visualizer::drawMap(classifier, config);

    std::cout << "\nPress Enter to see decision boundary with training data...";
    std::cin.get();

    Visualizer::drawMapWithData(classifier, config);

    std::cout << "\nPress Enter to see training data scatter plot...";
    std::cin.get();

    Visualizer::drawDataPoints(dataset, config);

    std::cout << "\nPress Enter to see query point prediction...";
    std::cin.get();

    Point query1(10.0, 10.0);
    Visualizer::drawMapWithQuery(classifier, query1, config);

    std::cout << "\nPress Enter to see k-nearest neighbors...";
    std::cin.get();

    Point query2(7.0, 8.0);
    Visualizer::drawMapWithNeighbors(classifier, query2, config);

    std::cout << std::endl;
}

/**
 * @brief Demo of multi-class visualization
 */
void demoMultiClassVisualization() {
    std::cout << "=== Multi-Class Visualization Demo ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Generating 4-class dataset..." << std::endl;

    std::vector<ClusterConfig> configs = {
        ClusterConfig(3.0, 3.0, 0.8, 0, 40),    // Bottom-left
        ClusterConfig(12.0, 3.0, 0.8, 1, 40),   // Bottom-right
        ClusterConfig(3.0, 12.0, 0.8, 2, 40),   // Top-left
        ClusterConfig(12.0, 12.0, 0.8, 3, 40)   // Top-right
    };

    auto dataset = Dataset::generateClusters(configs, 42);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 50;
    config.k = 7;
    config.useColors = true;

    std::cout << "\nPress Enter to see 4-class decision boundaries...";
    std::cin.get();

    Visualizer::drawMapWithData(classifier, config);

    std::cout << std::endl;
}

/**
 * @brief Interactive visualization mode
 */
void interactiveVisualization() {
    std::cout << "=== Interactive Visualization ===" << std::endl;
    std::cout << std::endl;

    auto dataset = Dataset::generateSimpleDataset(80, 123);
    KNN classifier(dataset);

    VisualizerConfig config;
    config.gridSize = 40;
    config.k = 5;
    config.useColors = true;

    auto [minX, maxX, minY, maxY] = Dataset::getBounds(dataset);

    std::cout << "Dataset bounds: X[" << minX << ", " << maxX << "], Y[" << minY << ", " << maxY
              << "]" << std::endl;
    std::cout << "\nEnter query points to visualize predictions." << std::endl;
    std::cout << "Type 'exit' to quit.\n" << std::endl;

    std::string input;
    while (true) {
        std::cout << "Enter X Y coordinates (or 'exit'): ";
        std::getline(std::cin, input);

        if (input == "exit" || input == "quit" || input == "q") {
            break;
        }

        std::istringstream iss(input);
        double x, y;

        if (iss >> x >> y) {
            Point query(x, y);

            Visualizer::clearScreen();
            Visualizer::drawMapWithNeighbors(classifier, query, config);

            std::cout << "\n";
        } else {
            std::cout << "Invalid input. Please enter two numbers (x y)." << std::endl;
        }
    }

    std::cout << "\nExiting interactive mode." << std::endl;
}

int main() {
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        
        std::ios::sync_with_stdio(false);
        
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        GetConsoleMode(hOut, &dwMode);
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
        
        CONSOLE_FONT_INFOEX fontInfo = {};
        fontInfo.cbSize = sizeof(fontInfo);
        GetCurrentConsoleFontEx(hOut, FALSE, &fontInfo);
        wcscpy_s(fontInfo.FaceName, L"Consolas");
        SetCurrentConsoleFontEx(hOut, FALSE, &fontInfo);
    #endif

    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  KNN Classifier with Console Visualization     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    std::cout << "Select demo mode:" << std::endl;
    std::cout << "1. Full demo (all features)" << std::endl;
    std::cout << "2. Visualization demo" << std::endl;
    std::cout << "3. Interactive mode" << std::endl;
    std::cout << "4. Quick test" << std::endl;
    std::cout << "\nChoice (1-4): ";

    int choice;
    std::cin >> choice;
    std::cin.ignore();

    switch (choice) {
        case 1:
            testPointDistance();
            demoDatasetGeneration();
            demoKNNClassifier();
            demoDatasetUtilities();
            benchmarkKNN();
            demoVisualization();
            demoMultiClassVisualization();
            break;

        case 2:
            demoVisualization();
            demoMultiClassVisualization();
            break;

        case 3:
            interactiveVisualization();
            break;

        case 4:
            testPointDistance();
            {
                auto dataset = Dataset::generateSimpleDataset(50, 42);
                KNN classifier(dataset);
                VisualizerConfig config;
                config.gridSize = 35;
                config.k = 5;
                Visualizer::drawMapWithData(classifier, config);
            }
            break;

        default:
            std::cout << "Invalid choice. Running quick test..." << std::endl;
            testPointDistance();
            break;
    }

    std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  All demos completed successfully! ✓✓✓         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;

    return 0;
}