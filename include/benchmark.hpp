#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <knn.hpp>
#include <knn_optimized.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

/**
 * @brief Benchmark result
 */
struct BenchmarkResult {
    std::string name;
    double averageTimeMs;
    double minTimeMs;
    double maxTimeMs;
    double stdDevMs;
    size_t iterations;
    size_t dataSize;
};

/**
 * @brief Benchmarking utilities
 */
namespace benchmark {

/**
 * @brief Measure execution time of a function
 * @param func Function to benchmark
 * @param iterations Number of iterations
 * @return BenchmarkResult with timing statistics
 */
template <typename Func>
BenchmarkResult measure(const std::string& name, Func func, size_t iterations = 100,
                       size_t dataSize = 0);

/**
 * @brief Compare standard KNN vs optimized KNN
 */
void compareKNNImplementations(const std::vector<Point>& trainData,
                               const std::vector<Point>& testData, int k);

/**
 * @brief Benchmark KD-tree vs linear search
 */
void benchmarkKDTree(const std::vector<Point>& data, int k, size_t numQueries);

/**
 * @brief Benchmark parallel vs sequential batch prediction
 */
void benchmarkParallelization(const std::vector<Point>& trainData,
                               const std::vector<Point>& testData, int k);

/**
 * @brief Benchmark different weighting strategies
 */
void benchmarkWeightStrategies(const std::vector<Point>& trainData,
                                const std::vector<Point>& testData, int k);

/**
 * @brief Full benchmark suite
 */
void runFullSuite(int datasetSize = 1000, int testSize = 200);

/**
 * @brief Print benchmark result
 */
void printResult(const BenchmarkResult& result);

/**
 * @brief Save benchmark results to CSV
 */
bool saveResultsToCSV(const std::vector<BenchmarkResult>& results,
                      const std::string& filename);

}  // namespace Benchmark

template <typename Func>
BenchmarkResult benchmark::measure(
    const std::string& name, Func func,
    size_t iterations, size_t dataSize) {
    std::vector<double> times;
    times.reserve(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration
            = std::chrono::duration_cast<
                std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);
    }

    double sum = 0.0;
    double minTime = times[0];
    double maxTime = times[0];

    for (double time : times) {
        sum += time;
        minTime = std::min(minTime, time);
        maxTime = std::max(maxTime, time);
    }

    double mean = sum / times.size();

    double variance = 0.0;
    for (double time : times)
        variance += (time - mean) * (time - mean);
    double stdDev = std::sqrt(variance / times.size());

    return BenchmarkResult{
        name, mean, minTime, maxTime,
        stdDev, iterations, dataSize};
}

#endif  // BENCHMARK_HPP
