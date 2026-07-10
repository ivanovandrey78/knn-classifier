#include <knn_optimized.hpp>
#include <dataset.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

int KnnOptimized::predictWeighted(const Point& query, int k) const {
    auto neighbors = getKNearest(query, k);
    if (neighbors.empty()) return 0;

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
                weight = std::exp(
                    -(dist * dist)
                    / (2.0 * gaussianSigma * gaussianSigma));
                break;
        }

        votes[neighbor.label] += weight;
    }

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

int KnnOptimized::predict(const Point& query, int k) const {
    if (weightStrategy != WeightStrategy::UNIFORM)
        return predictWeighted(query, k);

    if (use_kd_tree && has_kd_tree()) {
        auto neighbors = kd_tree.kNearestNeighbors(query, k);

        std::map<int, int> votes;
        for (const auto& [dist, point] : neighbors) {
            votes[point.label]++;
        }

        int bestLabel = votes.begin()->first;
        int maxVotes = votes.begin()->second;

        for (const auto& [label, count] : votes) {
            if (count > maxVotes
                || (count == maxVotes && label < bestLabel)) {
                maxVotes = count;
                bestLabel = label;
            }
        }

        return bestLabel;
    }

    return Knn::predict(query, k);
}

std::vector<Neighbor> KnnOptimized::getKNearest(
    const Point& query, int k) const {
    if (use_kd_tree && has_kd_tree()) {
        auto kdNeighbors = kd_tree.kNearestNeighbors(query, k);
        std::vector<Neighbor> result;
        result.reserve(kdNeighbors.size());

        for (const auto& [dist, point] : kdNeighbors) {
            result.emplace_back(dist, point.label, point);
        }

        return result;
    }

    return Knn::getKNearest(query, k);
}

std::vector<int> KnnOptimized::predictBatchParallel(
    const std::vector<Point>& queries, int k,
    int numThreads) const {
    if (queries.empty()) return {};

    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;
    }

    size_t nThreads = std::min<size_t>(numThreads, queries.size());

    std::vector<int> results(queries.size());

    if (nThreads == 1) return predictBatch(queries, k);

    auto worker = [this, &queries, k, &results](
        size_t start, size_t end) {
        for (size_t i = start; i < end; ++i)
            results[i] = predict(queries[i], k);
    };

    std::vector<std::thread> threads;
    size_t chunkSize = queries.size() / nThreads;
    size_t remainder = queries.size() % nThreads;

    size_t start = 0;
    for (size_t i = 0; i < nThreads; ++i) {
        size_t end = start + chunkSize
            + (i < remainder ? 1 : 0);
        threads.emplace_back(worker, start, end);
        start = end;
    }

    for (auto& thread : threads)
        thread.join();

    return results;
}

std::vector<Point> KnnOptimized::rangeQuery(
    double minX, double maxX, double minY, double maxY) const {
    if (use_kd_tree && has_kd_tree())
        return kd_tree.rangeQuery(minX, maxX, minY, maxY);

    std::vector<Point> results;
    for (const auto& point : trainingData) {
        if (point.x >= minX && point.x <= maxX
            && point.y >= minY && point.y <= maxY) {
            results.push_back(point);
        }
    }
    return results;
}

namespace cross_validation {

std::vector<double> kFoldCV(
    const std::vector<Point>& data, int folds, int knn,
    unsigned int seed) {
    if (folds <= 1 || data.empty()) return {};

    std::vector<Point> shuffled = data;
    std::mt19937 gen(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), gen);

    size_t foldSize = data.size() / folds;
    std::vector<double> accuracies;

    for (int fold = 0; fold < folds; ++fold) {
        size_t valStart = fold * foldSize;
        size_t valEnd = (fold == folds - 1)
            ? data.size() : (fold + 1) * foldSize;

        std::vector<Point> train, val;

        for (size_t i = 0; i < data.size(); ++i) {
            if (i >= valStart && i < valEnd) {
                val.push_back(shuffled[i]);
            } else {
                train.push_back(shuffled[i]);
            }
        }

        KnnOptimized classifier(train, true);

        std::vector<int> predictions;
        std::vector<int> groundTruth;

        for (const auto& point : val) {
            predictions.push_back(classifier.predict(point, knn));
            groundTruth.push_back(point.label);
        }

        double accuracy = dataset::calculateAccuracy(
            predictions, groundTruth);
        accuracies.push_back(accuracy);
    }

    return accuracies;
}

std::pair<int, double> findOptimalK(
    const std::vector<Point>& data, int maxK, int folds) {
    int bestK = 1;
    double bestAccuracy = 0.0;

    for (int k = 1; k <= maxK; ++k) {
        auto accuracies = kFoldCV(data, folds, k, 42);

        double meanAccuracy = std::accumulate(
            accuracies.begin(), accuracies.end(), 0.0)
            / accuracies.size();

        if (meanAccuracy > bestAccuracy) {
            bestAccuracy = meanAccuracy;
            bestK = k;
        }
    }

    return {bestK, bestAccuracy};
}

}  // namespace cross_validation
