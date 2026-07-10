#ifndef KNN_HPP
#define KNN_HPP

#include <point.hpp>
#include <vector>

/**
 * @brief Neighbor information (distance and label)
 */
struct Neighbor {
    double distance;
    int label;
    Point point;

    Neighbor(double d, int l, const Point& p) : distance(d), label(l), point(p) {}

    bool operator<(const Neighbor& other) const { return distance < other.distance; }
};

/**
 * @brief K-Nearest Neighbors classifier
 */
class KNN {
private:
    std::vector<Point> trainingData;

public:
    KNN() = default;
    explicit KNN(const std::vector<Point>& data) : trainingData(data) {}

    /**
     * @brief Set training data
     * @param data Vector of training points
     */
    void setTrainingData(const std::vector<Point>& data);

    /**
     * @brief Predict class label for a query point
     * @param query The point to classify
     * @param k Number of nearest neighbors to consider
     * @return Predicted class label
     * @throws std::runtime_error if training data is empty
     * @throws std::invalid_argument if k is invalid
     */
    virtual int predict(const Point& query, int k) const;

    /**
     * @brief Get k nearest neighbors for a query point
     * @param query The point to find neighbors for
     * @param k Number of neighbors to find
     * @return Vector of k nearest neighbors
     */
    virtual std::vector<Neighbor> getKNearest(const Point& query, int k) const;

    /**
     * @brief Predict with confidence score
     * @param query The point to classify
     * @param k Number of nearest neighbors
     * @return Pair of (predicted label, confidence 0-1)
     */
    std::pair<int, double> predictWithConfidence(const Point& query, int k) const;

    /**
     * @brief Batch prediction for multiple points
     * @param queries Vector of points to classify
     * @param k Number of nearest neighbors
     * @return Vector of predicted labels
     */
    std::vector<int> predictBatch(const std::vector<Point>& queries, int k) const;

    /**
     * @brief Get the training data
     * @return Reference to training data vector
     */
    const std::vector<Point>& getTrainingData() const { return trainingData; }

    /**
     * @brief Get number of training points
     * @return Size of training data
     */
    size_t size() const { return trainingData.size(); }

    /**
     * @brief Check if classifier has training data
     * @return true if training data is not empty
     */
    bool isReady() const { return !trainingData.empty(); }
};

#endif  // KNN_HPP
