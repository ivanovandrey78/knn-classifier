#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <point.hpp>
#include <vector>
#include <memory>
#include <algorithm>

/**
 * @brief KD-Tree node for 2D points
 */
struct KDNode {
    Point point;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
    int axis;  // 0 for x, 1 for y

    KDNode(const Point& p, int ax) : point(p), axis(ax) {}
};

/**
 * @brief KD-Tree for efficient nearest neighbor search
 */
class KDTree {
private:
    std::unique_ptr<KDNode> root;
    size_t treeSize;

    /**
     * @brief Recursively build KD-tree
     */
    std::unique_ptr<KDNode> buildTree(std::vector<Point>& points, int depth, size_t start,
                                      size_t end);

    /**
     * @brief Recursively search for k-nearest neighbors
     */
    void searchKNearest(const KDNode* node, const Point& target, int k,
                        std::vector<std::pair<double, Point>>& nearest) const;

    /**
     * @brief Search in a range
     */
    void rangeSearch(const KDNode* node, double minX, double maxX, double minY, double maxY,
                     std::vector<Point>& results) const;

public:
    KDTree() : treeSize(0) {}

    /**
     * @brief Build tree from points
     * @param points Vector of points (will be modified)
     */
    void build(std::vector<Point> points);

    /**
     * @brief Find k nearest neighbors
     * @param target Query point
     * @param k Number of neighbors
     * @return Vector of (distance, point) pairs, sorted by distance
     */
    std::vector<std::pair<double, Point>> kNearestNeighbors(const Point& target, int k) const;

    /**
     * @brief Range query - find all points in rectangle
     * @param minX Minimum x coordinate
     * @param maxX Maximum x coordinate
     * @param minY Minimum y coordinate
     * @param maxY Maximum y coordinate
     * @return Vector of points in range
     */
    std::vector<Point> rangeQuery(double minX, double maxX, double minY, double maxY) const;

    /**
     * @brief Get size of tree
     */
    size_t size() const { return treeSize; }

    /**
     * @brief Check if tree is empty
     */
    bool empty() const { return treeSize == 0; }
};

#endif  // KDTREE_HPP
