#include <knn.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>

void KNN::setTrainingData(const std::vector<Point>& data) {
    trainingData = data;
}

int KNN::predict(const Point& query, int k) const {
    if (trainingData.empty()) {
        throw std::runtime_error("Training data is empty. Call setTrainingData() first.");
    }

    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    if (k > static_cast<int>(trainingData.size())) {
        throw std::invalid_argument("k cannot be larger than training data size");
    }

    // Create vector of (distance, label) pairs
    std::vector<std::pair<double, int>> distances;
    distances.reserve(trainingData.size());

    for (const auto& point : trainingData) {
        double dist = query.distanceTo(point);
        distances.emplace_back(dist, point.label);
    }

    // Partial sort to get k nearest neighbors (more efficient than full sort)
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    // Count votes for each label among k nearest neighbors
    std::map<int, int> votes;
    for (int i = 0; i < k; ++i) {
        votes[distances[i].second]++;
    }

    // Find label with maximum votes
    int predictedLabel = votes.begin()->first;
    int maxVotes = votes.begin()->second;

    for (const auto& [label, count] : votes) {
        if (count > maxVotes) {
            maxVotes = count;
            predictedLabel = label;
        } else if (count == maxVotes && label < predictedLabel) {
            // Tie-breaking: choose smaller label
            predictedLabel = label;
        }
    }

    return predictedLabel;
}

std::vector<Neighbor> KNN::getKNearest(const Point& query, int k) const {
    if (trainingData.empty()) {
        throw std::runtime_error("Training data is empty");
    }

    if (k <= 0 || k > static_cast<int>(trainingData.size())) {
        throw std::invalid_argument("Invalid k value");
    }

    std::vector<Neighbor> neighbors;
    neighbors.reserve(trainingData.size());

    for (const auto& point : trainingData) {
        double dist = query.distanceTo(point);
        neighbors.emplace_back(dist, point.label, point);
    }

    // Partial sort to get k nearest
    std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end());

    // Keep only k nearest
    neighbors.resize(k);
    return neighbors;
}

std::pair<int, double> KNN::predictWithConfidence(const Point& query, int k) const {
    auto neighbors = getKNearest(query, k);

    // Count votes
    std::map<int, int> votes;
    for (const auto& neighbor : neighbors) {
        votes[neighbor.label]++;
    }

    // Find winner
    int predictedLabel = votes.begin()->first;
    int maxVotes = votes.begin()->second;

    for (const auto& [label, count] : votes) {
        if (count > maxVotes) {
            maxVotes = count;
            predictedLabel = label;
        } else if (count == maxVotes && label < predictedLabel) {
            predictedLabel = label;
        }
    }

    // Confidence is the ratio of votes for winning label
    double confidence = static_cast<double>(maxVotes) / k;

    return {predictedLabel, confidence};
}

std::vector<int> KNN::predictBatch(const std::vector<Point>& queries, int k) const {
    std::vector<int> predictions;
    predictions.reserve(queries.size());

    for (const auto& query : queries) {
        predictions.push_back(predict(query, k));
    }

    return predictions;
}
