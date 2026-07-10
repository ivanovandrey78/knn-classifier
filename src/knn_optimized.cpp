#include <knn_optimized.hpp>
#include <dataset.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

void KNNOptimized::buildKDTree() {
    if (trainingData.empty()) {
        return;
    }
    kdTree.build(trainingData);
    useKDTree = true;
}

void KNNOptimized::setWeightStrategy(WeightStrategy strategy, double sigma) {
    weightStrategy = strategy;
    gaussianSigma = sigma;
}

int KNNOptimized::predictWeighted(const Point& query, int k) const {
    auto neighbors = getKNearest(query, k);
    if (neighbors.empty()) {
        return 0;
    }

    // Calculate weighted votes
    std::map<int, double> votes;

    for (const auto& neighbor : neighbors) {
        double weight = 1.0;
        double dist = neighbor.distance;

        switch (weightStrategy) {
            case WeightStrategy::UNIFORM:
                weight = 1.0;
                break;

            case WeightStrategy::INVERSE_DISTANCE:
                weight = dist > 1e-9 ? 1.0 / dist : 1e9;
                break;

            case WeightStrategy::SQUARED_INVERSE:
                weight = dist > 1e-9 ? 1.0 / (dist * dist) : 1e9;
                break;

            case WeightStrategy::GAUSSIAN:
                weight = std::exp(-(dist * dist) / (2.0 * gaussianSigma * gaussianSigma));
                break;
        }

        votes[neighbor.label] += weight;
    }

    // Find label with maximum weighted vote
    int bestLabel = votes.begin()->first;
    double maxVote = votes.begin()->second;

    for (const auto& [label, vote] : votes) {
        if (vote > maxVote || (vote == maxVote && label < bestLabel)) {
            maxVote = vote;
            bestLabel = label;
        }
    }

    return bestLabel;
}

int KNNOptimized::predict(const Point& query, int k) const {
    if (weightStrategy != WeightStrategy::UNIFORM) {
        return predictWeighted(query, k);
    }

    if (useKDTree && hasKDTree()) {
        // Use KD-tree for fast search
        auto neighbors = kdTree.kNearestNeighbors(query, k);

        std::map<int, int> votes;
        for (const auto& [dist, point] : neighbors) {
            votes[point.label]++;
        }

        int bestLabel = votes.begin()->first;
        int maxVotes = votes.begin()->second;

        for (const auto& [label, count] : votes) {
            if (count > maxVotes || (count == maxVotes && label < bestLabel)) {
                maxVotes = count;
                bestLabel = label;
            }
        }

        return bestLabel;
    } else {
        // Fall back to base implementation
        return KNN::predict(query, k);
    }
}

std::vector<Neighbor> KNNOptimized::getKNearest(const Point& query, int k) const {
    if (useKDTree && hasKDTree()) {
        auto kdNeighbors = kdTree.kNearestNeighbors(query, k);
        std::vector<Neighbor> result;
        result.reserve(kdNeighbors.size());

        for (const auto& [dist, point] : kdNeighbors) {
            result.emplace_back(dist, point.label, point);
        }

        return result;
    } else {
        return KNN::getKNearest(query, k);
    }
}

void KNNOptimized::predictBatchWorker(const KNNOptimized* classifier,
                                     const std::vector<Point>& queries, size_t start, size_t end,
                                     int k, std::vector<int>& results) {
    for (size_t i = start; i < end; ++i) {
        results[i] = classifier->predict(queries[i], k);
    }
}

std::vector<int> KNNOptimized::predictBatchParallel(const std::vector<Point>& queries, int k,
                                                    int numThreads) const {
    if (queries.empty()) {
        return {};
    }

    // Determine number of threads
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;  // Fallback
    }

    // Don't use more threads than queries
    numThreads = std::min(numThreads, static_cast<int>(queries.size()));

    std::vector<int> results(queries.size());

    if (numThreads == 1) {
        // Single-threaded
        return predictBatch(queries, k);
    }

    // Parallel execution
    std::vector<std::thread> threads;
    size_t chunkSize = queries.size() / numThreads;
    size_t remainder = queries.size() % numThreads;

    size_t start = 0;
    for (int i = 0; i < numThreads; ++i) {
        size_t end = start + chunkSize + (i < static_cast<int>(remainder) ? 1 : 0);

        threads.emplace_back(predictBatchWorker, this, std::cref(queries), start, end, k,
                            std::ref(results));

        start = end;
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return results;
}

std::vector<Point> KNNOptimized::rangeQuery(double minX, double maxX, double minY,
                                            double maxY) const {
    if (useKDTree && hasKDTree()) {
        return kdTree.rangeQuery(minX, maxX, minY, maxY);
    }

    // Fallback to linear search
    std::vector<Point> results;
    for (const auto& point : trainingData) {
        if (point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY) {
            results.push_back(point);
        }
    }
    return results;
}

namespace CrossValidation {

std::vector<double> kFoldCV(const std::vector<Point>& data, int folds, int knn,
                            unsigned int seed) {
    if (folds <= 1 || data.empty()) {
        return {};
    }

    // Shuffle data
    std::vector<Point> shuffled = data;
    std::mt19937 gen(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), gen);

    size_t foldSize = data.size() / folds;
    std::vector<double> accuracies;

    for (int fold = 0; fold < folds; ++fold) {
        // Split into train and validation
        size_t valStart = fold * foldSize;
        size_t valEnd = (fold == folds - 1) ? data.size() : (fold + 1) * foldSize;

        std::vector<Point> train, val;

        for (size_t i = 0; i < data.size(); ++i) {
            if (i >= valStart && i < valEnd) {
                val.push_back(shuffled[i]);
            } else {
                train.push_back(shuffled[i]);
            }
        }

        // Train and evaluate
        KNNOptimized classifier(train, true);  // Build KD-tree

        std::vector<int> predictions;
        std::vector<int> groundTruth;

        for (const auto& point : val) {
            predictions.push_back(classifier.predict(point, knn));
            groundTruth.push_back(point.label);
        }

        double accuracy = Dataset::calculateAccuracy(predictions, groundTruth);
        accuracies.push_back(accuracy);
    }

    return accuracies;
}

std::pair<int, double> findOptimalK(const std::vector<Point>& data, int maxK, int folds) {
    int bestK = 1;
    double bestAccuracy = 0.0;

    for (int k = 1; k <= maxK; ++k) {
        auto accuracies = kFoldCV(data, folds, k, 42);

        // Calculate mean accuracy
        double meanAccuracy =
            std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / accuracies.size();

        if (meanAccuracy > bestAccuracy) {
            bestAccuracy = meanAccuracy;
            bestK = k;
        }
    }

    return {bestK, bestAccuracy};
}

}  // namespace CrossValidation
