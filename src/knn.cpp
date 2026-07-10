#include <knn.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>

int KNN::predict(const Point& query, int k) const {
    if (trainingData.empty())
        throw std::runtime_error(
            "Training data is empty. Call setTrainingData() first.");

    if (k <= 0)
        throw std::invalid_argument("k must be positive");

    if (k > static_cast<int>(trainingData.size()))
        throw std::invalid_argument(
            "k cannot be larger than training data size");

    std::vector<std::pair<double, int>> distances;
    distances.reserve(trainingData.size());

    for (const auto& point : trainingData) {
        double dist = query.distanceTo(point);
        distances.emplace_back(dist, point.label);
    }

    std::partial_sort(distances.begin(), distances.begin() + k,
                      distances.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

    std::map<int, int> votes;
    for (int i = 0; i < k; ++i)
        votes[distances[i].second]++;

    int predicted_label = votes.begin()->first;
    int max_votes = votes.begin()->second;

    for (const auto& [label, count] : votes) {
        if (count > max_votes) {
            max_votes = count;
            predicted_label = label;
        } else if (count == max_votes && label < predicted_label) {
            predicted_label = label;
        }
    }

    return predicted_label;
}

std::vector<Neighbor> KNN::getKNearest(
    const Point& query, int k) const {
    if (trainingData.empty())
        throw std::runtime_error("Training data is empty");

    if (k <= 0 || k > static_cast<int>(trainingData.size()))
        throw std::invalid_argument("Invalid k value");

    std::vector<Neighbor> neighbors;
    neighbors.reserve(trainingData.size());

    for (const auto& point : trainingData) {
        double dist = query.distanceTo(point);
        neighbors.emplace_back(dist, point.label, point);
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + k,
                      neighbors.end());

    neighbors.resize(k);
    return neighbors;
}

std::pair<int, double> KNN::predictWithConfidence(
    const Point& query, int k) const {
    auto neighbors = getKNearest(query, k);

    std::map<int, int> votes;
    for (const auto& neighbor : neighbors)
        votes[neighbor.label]++;

    int predicted_label = votes.begin()->first;
    int max_votes = votes.begin()->second;

    for (const auto& [label, count] : votes) {
        if (count > max_votes) {
            max_votes = count;
            predicted_label = label;
        } else if (count == max_votes && label < predicted_label) {
            predicted_label = label;
        }
    }

    double confidence = static_cast<double>(max_votes) / k;
    return {predicted_label, confidence};
}

std::vector<int> KNN::predictBatch(
    const std::vector<Point>& queries, int k) const {
    std::vector<int> predictions;
    predictions.reserve(queries.size());

    for (const auto& query : queries)
        predictions.push_back(predict(query, k));

    return predictions;
}
