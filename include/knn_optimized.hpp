#ifndef KNN_OPTIMIZED_HPP
#define KNN_OPTIMIZED_HPP

#include <knn.hpp>
#include <kdtree.hpp>
#include <vector>
#include <thread>
#include <future>

enum class WeightStrategy {
    UNIFORM,
    INVERSE_DISTANCE,
    SQUARED_INVERSE,
    GAUSSIAN
};

class KnnOptimized : public Knn {
private:
    KdTree kd_tree;
    bool use_kd_tree;
    WeightStrategy weightStrategy;
    double gaussianSigma;

    int predictWeighted(const Point& query, int k) const;

public:
    KnnOptimized()
        : use_kd_tree(false)
        , weightStrategy(WeightStrategy::UNIFORM)
        , gaussianSigma(1.0) {}

    explicit KnnOptimized(
        const std::vector<Point>& data, bool buildTree = true)
        : Knn(data)
        , use_kd_tree(buildTree)
        , weightStrategy(WeightStrategy::UNIFORM)
        , gaussianSigma(1.0) {
        if (buildTree) buildKdTree();
    }

    void buildKdTree() {
        if (trainingData.empty()) return;
        kd_tree.build(trainingData);
        use_kd_tree = true;
    }

    void setWeightStrategy(
        WeightStrategy strategy, double sigma = 1.0) {
        weightStrategy = strategy;
        gaussianSigma = sigma;
    }

    int predict(const Point& query, int k) const override;

    std::vector<int> predictBatchParallel(
        const std::vector<Point>& queries, int k,
        int numThreads = 0) const;

    std::vector<Neighbor> getKNearest(
        const Point& query, int k) const override;

    std::vector<Point> rangeQuery(
        double minX, double maxX, double minY, double maxY) const;

    bool has_kd_tree() const {
        return use_kd_tree && !kd_tree.empty();
    }
};

namespace cross_validation {

std::vector<double> kFoldCV(
    const std::vector<Point>& data, int folds, int knn,
    unsigned int seed = 42);

std::pair<int, double> findOptimalK(
    const std::vector<Point>& data, int maxK = 20,
    int folds = 5);

}  // namespace cross_validation

#endif  // KNN_OPTIMIZED_HPP
