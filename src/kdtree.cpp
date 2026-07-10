#include <kdtree.hpp>
#include <algorithm>
#include <queue>
#include <cmath>

std::unique_ptr<KdNode> KdTree::buildTree(
    std::vector<Point>& points, int depth, size_t start, size_t end) {
    if (start >= end) return nullptr;

    int axis = depth % 2;
    size_t mid = start + (end - start) / 2;

    std::nth_element(points.begin() + start, points.begin() + mid,
                     points.begin() + end,
                     [axis](const Point& a, const Point& b) {
                         return axis == 0 ? a.x < b.x : a.y < b.y;
                     });

    auto node = std::make_unique<KdNode>(points[mid], axis);
    node->left = buildTree(points, depth + 1, start, mid);
    node->right = buildTree(points, depth + 1, mid + 1, end);

    return node;
}

void KdTree::build(std::vector<Point> points) {
    treeSize = points.size();
    if (treeSize == 0) {
        root = nullptr;
        return;
    }
    root = buildTree(points, 0, 0, points.size());
}

void KdTree::searchKNearest(
    const KdNode* node, const Point& target, int k,
    std::vector<std::pair<double, Point>>& nearest) const {
    if (!node) return;

    double dist = target.distanceTo(node->point);
    auto cmp = [](const auto& a, const auto& b) {
        return a.first < b.first;
    };

    if (nearest.size() < static_cast<size_t>(k)) {
        nearest.push_back({dist, node->point});
        std::push_heap(nearest.begin(), nearest.end(), cmp);
    } else if (dist < nearest.front().first) {
        std::pop_heap(nearest.begin(), nearest.end(), cmp);
        nearest.back() = {dist, node->point};
        std::push_heap(nearest.begin(), nearest.end(), cmp);
    }

    double diff = node->axis == 0
        ? target.x - node->point.x
        : target.y - node->point.y;
    KdNode* nearSide = diff < 0 ? node->left.get() : node->right.get();
    KdNode* farSide = diff < 0 ? node->right.get() : node->left.get();

    searchKNearest(nearSide, target, k, nearest);

    if (nearest.size() < static_cast<size_t>(k)
        || std::abs(diff) < nearest.front().first) {
        searchKNearest(farSide, target, k, nearest);
    }
}

std::vector<std::pair<double, Point>> KdTree::kNearestNeighbors(
    const Point& target, int k) const {
    if (!root || k <= 0) return {};

    std::vector<std::pair<double, Point>> nearest;
    nearest.reserve(k);

    searchKNearest(root.get(), target, k, nearest);

    std::sort(nearest.begin(), nearest.end(),
              [](const auto& a, const auto& b) {
                  return a.first < b.first;
              });

    return nearest;
}

void KdTree::rangeSearch(
    const KdNode* node, double minX, double maxX,
    double minY, double maxY,
    std::vector<Point>& results) const {
    if (!node) return;

    if (node->point.x >= minX && node->point.x <= maxX
        && node->point.y >= minY && node->point.y <= maxY) {
        results.push_back(node->point);
    }

    if (node->axis == 0) {
        if (minX <= node->point.x)
            rangeSearch(node->left.get(), minX, maxX, minY, maxY, results);
        if (maxX >= node->point.x)
            rangeSearch(node->right.get(), minX, maxX, minY, maxY, results);
    } else {
        if (minY <= node->point.y)
            rangeSearch(node->left.get(), minX, maxX, minY, maxY, results);
        if (maxY >= node->point.y)
            rangeSearch(node->right.get(), minX, maxX, minY, maxY, results);
    }
}

std::vector<Point> KdTree::rangeQuery(
    double minX, double maxX, double minY, double maxY) const {
    std::vector<Point> results;
    if (root)
        rangeSearch(root.get(), minX, maxX, minY, maxY, results);
    return results;
}
