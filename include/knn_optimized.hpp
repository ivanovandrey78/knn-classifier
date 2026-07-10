#ifndef KNN_OPTIMIZED_HPP
#define KNN_OPTIMIZED_HPP

#include <knn.hpp>
#include <kdtree.hpp>
#include <vector>
#include <thread>
#include <future>

/**
 * @brief Distance weighting strategy
 */
enum class WeightStrategy {
    UNIFORM,           // All neighbors equal weight
    INVERSE_DISTANCE,  // 1/distance weighting
    SQUARED_INVERSE,   // 1/distance^2 weighting
    GAUSSIAN           // Gaussian kernel weighting
};

/**
 * @brief Optimized KNN classifier with KD-tree and parallelization
 */
class KNNOptimized : public KNN {
private:
    KDTree kdTree;
    bool useKDTree;
    WeightStrategy weightStrategy;
    double gaussianSigma;

    /**
     * @brief Calculate weighted vote
     */
    int predictWeighted(const Point& query, int k) const;

    /**
     * @brief Worker function for parallel batch prediction
     */
    static void predictBatchWorker(const KNNOptimized* classifier, 
                                   const std::vector<Point>& queries,
                                   size_t start, size_t end, int k,
                                   std::vector<int>& results);

public:
    KNNOptimized() : useKDTree(false), weightStrategy(WeightStrategy::UNIFORM), 
                     gaussianSigma(1.0) {}

    explicit KNNOptimized(const std::vector<Point>& data, bool buildTree = true)
        : KNN(data), useKDTree(buildTree), weightStrategy(WeightStrategy::UNIFORM),
          gaussianSigma(1.0) {
        if (buildTree) {
            buildKDTree();
        }
    }

    /**
     * @brief Build KD-tree index
     */
    void buildKDTree();

    /**
     * @brief Set weight strategy
     */
    void setWeightStrategy(WeightStrategy strategy, double sigma = 1.0);

    /**
     * @brief Predict using KD-tree (if available)
     */
    int predict(const Point& query, int k) const override;

    /**
     * @brief Parallel batch prediction
     * @param queries Vector of points to classify
     * @param k Number of neighbors
     * @param numThreads Number of threads (0 = auto)
     * @return Vector of predictions
     */
    std::vector<int> predictBatchParallel(const std::vector<Point>& queries, int k,
                                          int numThreads = 0) const;

    /**
     * @brief Get k nearest neighbors using KD-tree
     */
    std::vector<Neighbor> getKNearest(const Point& query, int k) const override;

    /**
     * @brief Range query using KD-tree
     */
    std::vector<Point> rangeQuery(double minX, double maxX, double minY, double maxY) const;

    /**
     * @brief Check if KD-tree is built
     */
    bool hasKDTree() const { return useKDTree && !kdTree.empty(); }
};

/**
 * @brief Cross-validation utilities
 */
namespace CrossValidation {

/**
 * @brief Perform k-fold cross-validation
 * @param data Full dataset
 * @param k Number of folds
 * @param knn K for KNN classifier
 * @param seed Random seed
 * @return Vector of accuracy scores for each fold
 */
std::vector<double> kFoldCV(const std::vector<Point>& data, int folds, int knn,
                            unsigned int seed = 42);

/**
 * @brief Find optimal k using cross-validation
 * @param data Full dataset
 * @param maxK Maximum k to test
 * @param folds Number of CV folds
 * @return Pair of (best k, best accuracy)
 */
std::pair<int, double> findOptimalK(const std::vector<Point>& data, int maxK = 20,
                                    int folds = 5);

}  // namespace CrossValidation

#endif  // KNN_OPTIMIZED_HPP
