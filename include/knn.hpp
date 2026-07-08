#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include "point.hpp"

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
     */
    int predict(const Point& query, int k) const;

    /**
     * @brief Get the training data
     * @return Reference to training data vector
     */
    const std::vector<Point>& getTrainingData() const { return trainingData; }
};

#endif  // KNN_HPP
