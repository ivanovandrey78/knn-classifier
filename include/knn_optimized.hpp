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

class KNNOptimized : public KNN {
private:
    KDTree kdTree;
    bool useKDTree;
    WeightStrategy weightStrategy;
    double gaussianSigma;

    int predictWeighted(const Point& query, int k) const;

public:
    KNNOptimized()
        : useKDTree(false)
        , weightStrategy(WeightStrategy::UNIFORM)
        , gaussianSigma(1.0) {}

    explicit KNNOptimized(
        const std::vector<Point>& data, bool buildTree = true)
        : KNN(data)
        , useKDTree(buildTree)
        , weightStrategy(WeightStrategy::UNIFORM)
        , gaussianSigma(1.0) {
        if (buildTree) buildKDTree();
    }

    void buildKDTree() {
        if (trainingData.empty()) return;
        kdTree.build(trainingData);
        useKDTree = true;
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

    bool hasKDTree() const {
        return useKDTree && !kdTree.empty();
    }
};

namespace CrossValidation {

std::vector<double> kFoldCV(
    const std::vector<Point>& data, int folds, int knn,
    unsigned int seed = 42);

std::pair<int, double> findOptimalK(
    const std::vector<Point>& data, int maxK = 20,
    int folds = 5);

}  // namespace CrossValidation

#endif  // KNN_OPTIMIZED_HPP
