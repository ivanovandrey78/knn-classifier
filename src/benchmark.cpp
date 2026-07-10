#include <benchmark.hpp>
#include <dataset.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace Benchmark {

void printResult(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::setw(40) << std::left << result.name;
    std::cout << " | Avg: " << std::setw(8) << std::right
              << result.averageTimeMs << " ms";
    std::cout << " | Min: " << std::setw(8) << result.minTimeMs
              << " ms";
    std::cout << " | Max: " << std::setw(8) << result.maxTimeMs
              << " ms";
    std::cout << " | StdDev: " << std::setw(6) << result.stdDevMs
              << " ms";
    if (result.dataSize > 0)
        std::cout << " | Size: " << result.dataSize;
    std::cout << std::endl;
}

void compareKNNImplementations(
    const std::vector<Point>& trainData,
    const std::vector<Point>& testData, int k) {
    std::cout << "\n=== KNN Implementation Comparison ==="
              << std::endl;
    std::cout << "Training size: " << trainData.size()
              << ", Test size: " << testData.size()
              << ", k: " << k << std::endl;
    std::cout << std::endl;

    KNN standardKNN(trainData);
    auto standardResult = measure(
        "Standard KNN",
        [&]() { standardKNN.predictBatch(testData, k); },
        10, trainData.size());
    printResult(standardResult);

    KNNOptimized optimizedKNN(trainData, true);
    auto optimizedResult = measure(
        "Optimized KNN (KD-tree)",
        [&]() { optimizedKNN.predictBatch(testData, k); },
        10, trainData.size());
    printResult(optimizedResult);

    double speedup
        = standardResult.averageTimeMs
          / optimizedResult.averageTimeMs;
    std::cout << "\nSpeedup: " << std::fixed
              << std::setprecision(2) << speedup << "x"
              << std::endl;
}

void benchmarkKDTree(
    const std::vector<Point>& data, int k, size_t numQueries) {
    std::cout << "\n=== KD-Tree vs Linear Search ===" << std::endl;
    std::cout << "Data size: " << data.size()
              << ", Queries: " << numQueries
              << ", k: " << k << std::endl;
    std::cout << std::endl;

    auto [minX, maxX, minY, maxY] = Dataset::getBounds(data);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> distX(minX, maxX);
    std::uniform_real_distribution<> distY(minY, maxY);

    std::vector<Point> queries;
    for (size_t i = 0; i < numQueries; ++i)
        queries.emplace_back(distX(gen), distY(gen));

    KNN linearKNN(data);
    auto linearResult = measure(
        "Linear Search",
        [&]() {
            for (const auto& q : queries) linearKNN.predict(q, k);
        },
        10, data.size());
    printResult(linearResult);

    KNNOptimized kdTreeKNN(data, true);
    auto kdTreeResult = measure(
        "KD-Tree Search",
        [&]() {
            for (const auto& q : queries) kdTreeKNN.predict(q, k);
        },
        10, data.size());
    printResult(kdTreeResult);

    double speedup
        = linearResult.averageTimeMs
          / kdTreeResult.averageTimeMs;
    std::cout << "\nSpeedup: " << std::fixed
              << std::setprecision(2) << speedup << "x"
              << std::endl;
}

void benchmarkParallelization(
    const std::vector<Point>& trainData,
    const std::vector<Point>& testData, int k) {
    std::cout << "\n=== Parallel vs Sequential Prediction ==="
              << std::endl;
    std::cout << "Training size: " << trainData.size()
              << ", Test size: " << testData.size() << std::endl;
    std::cout << std::endl;

    KNNOptimized classifier(trainData, true);

    auto seqResult = measure(
        "Sequential",
        [&]() { classifier.predictBatch(testData, k); },
        10, testData.size());
    printResult(seqResult);

    std::vector<int> threadCounts = {2, 4, 8};
    for (int threads : threadCounts) {
        std::string name
            = "Parallel (" + std::to_string(threads) + " threads)";
        auto parResult = measure(
            name,
            [&]() {
                classifier.predictBatchParallel(testData, k, threads);
            },
            10, testData.size());
        printResult(parResult);

        double speedup
            = seqResult.averageTimeMs / parResult.averageTimeMs;
        std::cout << "    → Speedup: " << std::fixed
                  << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }
}

void benchmarkWeightStrategies(
    const std::vector<Point>& trainData,
    const std::vector<Point>& testData, int k) {
    std::cout << "\n=== Weight Strategy Comparison ==="
              << std::endl;
    std::cout << "Training size: " << trainData.size()
              << ", Test size: " << testData.size() << std::endl;
    std::cout << std::endl;

    struct Strategy {
        std::string name;
        WeightStrategy strategy;
        double sigma;
    };

    std::vector<Strategy> strategies = {
        {"Uniform", WeightStrategy::UNIFORM, 1.0},
        {"Inverse Distance", WeightStrategy::INVERSE_DISTANCE, 1.0},
        {"Squared Inverse", WeightStrategy::SQUARED_INVERSE, 1.0},
        {"Gaussian (σ=1.0)", WeightStrategy::GAUSSIAN, 1.0},
        {"Gaussian (σ=2.0)", WeightStrategy::GAUSSIAN, 2.0}};

    for (const auto& strat : strategies) {
        KNNOptimized classifier(trainData, true);
        classifier.setWeightStrategy(strat.strategy, strat.sigma);

        auto result = measure(
            strat.name,
            [&]() { classifier.predictBatch(testData, k); },
            10, trainData.size());
        printResult(result);

        auto predictions = classifier.predictBatch(testData, k);
        std::vector<int> groundTruth;
        for (const auto& p : testData)
            groundTruth.push_back(p.label);
        double accuracy = Dataset::calculateAccuracy(
            predictions, groundTruth);
        std::cout << "    → Accuracy: " << std::fixed
                  << std::setprecision(2) << accuracy << "%"
                  << std::endl;
    }
}

void runFullSuite(int datasetSize, int testSize) {
    std::cout << "\n╔════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║     COMPREHENSIVE BENCHMARK SUITE       ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════╝"
              << std::endl;

    std::vector<int> sizes = {100, 500, 1000, 2000, 5000};

    std::cout << "\n=== Dataset Size Scaling ===" << std::endl;
    std::cout << std::endl;

    for (int size : sizes) {
        if (size > datasetSize) continue;

        auto data = Dataset::generateSimpleDataset(size / 2, 42);
        auto [train, test]
            = Dataset::trainTestSplit(data, 0.8, true, 42);

        KNN standard(train);
        KNNOptimized optimized(train, true);

        Point query(10.0, 10.0);
        auto standardTime = measure(
            "Standard (size=" + std::to_string(size) + ")",
            [&]() { standard.predict(query, 5); }, 100, size);

        auto optimizedTime = measure(
            "KD-tree (size=" + std::to_string(size) + ")",
            [&]() { optimized.predict(query, 5); }, 100, size);

        printResult(standardTime);
        printResult(optimizedTime);

        double speedup
            = standardTime.averageTimeMs
              / optimizedTime.averageTimeMs;
        std::cout << "  → Speedup: " << std::fixed
                  << std::setprecision(2) << speedup << "x\n"
                  << std::endl;
    }

    auto fullData = Dataset::generateSimpleDataset(
        datasetSize / 2, 42);
    auto [train, test]
        = Dataset::trainTestSplit(fullData, 0.8, true, 42);

    compareKNNImplementations(train, test, 5);
    benchmarkKDTree(train, 5, testSize);
    benchmarkParallelization(train, test, 5);
    benchmarkWeightStrategies(train, test, 5);

    std::cout << "\n╔════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║   Benchmark suite completed! ✓✓✓       ║"
              << std::endl;
    std::cout << "╚════════════════════════════════════════╝"
              << std::endl;
}

bool saveResultsToCSV(
    const std::vector<BenchmarkResult>& results,
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    file << "Name,AverageTimeMs,MinTimeMs,MaxTimeMs,"
            "StdDevMs,Iterations,DataSize\n";

    for (const auto& result : results) {
        file << result.name << "," << result.averageTimeMs << ","
             << result.minTimeMs << "," << result.maxTimeMs << ","
             << result.stdDevMs << "," << result.iterations << ","
             << result.dataSize << "\n";
    }

    file.close();
    return true;
}

}  // namespace Benchmark
