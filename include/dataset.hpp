#ifndef DATASET_HPP
#define DATASET_HPP

#include <point.hpp>
#include <random>
#include <string>
#include <vector>

/**
 * @brief Configuration for cluster generation
 */
struct ClusterConfig {
    double centerX;
    double centerY;
    double stdDev;
    int label;
    int numPoints;

    ClusterConfig(double cx, double cy, double std, int lbl, int num)
        : centerX(cx), centerY(cy), stdDev(std), label(lbl), numPoints(num) {}
};

/**
 * @brief Dataset utilities for generation, manipulation and evaluation
 */
namespace Dataset {

/**
 * @brief Generate synthetic dataset with multiple clusters
 * @param configs Vector of cluster configurations
 * @param seed Random seed for reproducibility (0 = random seed)
 * @return Vector of generated points
 */
std::vector<Point> generateClusters(const std::vector<ClusterConfig>& configs,
                                    unsigned int seed = 0);

/**
 * @brief Generate simple two-class dataset
 * @param numPointsPerClass Number of points per class
 * @param seed Random seed for reproducibility (0 = random seed)
 * @return Vector of generated points
 */
std::vector<Point> generateSimpleDataset(int numPointsPerClass = 100, unsigned int seed = 0);

/**
 * @brief Split dataset into training and testing sets
 * @param data Full dataset
 * @param trainRatio Ratio of training data (0.0 to 1.0)
 * @param shuffle Whether to shuffle before splitting
 * @param seed Random seed for shuffling
 * @return Pair of (training, testing) datasets
 */
std::pair<std::vector<Point>, std::vector<Point>> trainTestSplit(const std::vector<Point>& data,
                                                                   double trainRatio = 0.8,
                                                                   bool shuffle = true,
                                                                   unsigned int seed = 0);

/**
 * @brief Calculate classification accuracy
 * @param predictions Predicted labels
 * @param groundTruth Actual labels
 * @return Accuracy as percentage (0.0 to 100.0)
 */
double calculateAccuracy(const std::vector<int>& predictions,
                         const std::vector<int>& groundTruth);

/**
 * @brief Get statistics about the dataset
 * @param data Dataset to analyze
 * @return String with dataset statistics
 */
std::string getDatasetStats(const std::vector<Point>& data);

/**
 * @brief Calculate confusion matrix
 * @param predictions Predicted labels
 * @param groundTruth Actual labels
 * @param numClasses Number of classes
 * @return 2D vector representing confusion matrix
 */
std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& predictions,
                                               const std::vector<int>& groundTruth,
                                               int numClasses);

/**
 * @brief Save dataset to CSV file
 * @param data Dataset to save
 * @param filename Output filename
 * @return true if successful
 */
bool saveToCSV(const std::vector<Point>& data, const std::string& filename);

/**
 * @brief Load dataset from CSV file
 * @param filename Input filename
 * @return Loaded dataset
 */
std::vector<Point> loadFromCSV(const std::string& filename);

/**
 * @brief Normalize dataset to [0, 1] range
 * @param data Dataset to normalize (modified in place)
 */
void normalize(std::vector<Point>& data);

/**
 * @brief Get bounds of the dataset
 * @param data Dataset
 * @return Tuple of (minX, maxX, minY, maxY)
 */
std::tuple<double, double, double, double> getBounds(const std::vector<Point>& data);

}  // namespace Dataset

#endif  // DATASET_HPP
