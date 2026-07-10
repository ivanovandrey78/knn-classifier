#include <kdtree.hpp>
#include <algorithm>
#include <queue>
#include <cmath>

std::unique_ptr<KDNode> KDTree::buildTree(std::vector<Point>& points, int depth, size_t start,
                                          size_t end) {
    if (start >= end) {
        return nullptr;
    }

    int axis = depth % 2;  // 0 = x, 1 = y
    size_t mid = start + (end - start) / 2;

    // Sort by current axis
    std::nth_element(points.begin() + start, points.begin() + mid, points.begin() + end,
                     [axis](const Point& a, const Point& b) {
                         return axis == 0 ? a.x < b.x : a.y < b.y;
                     });

    auto node = std::make_unique<KDNode>(points[mid], axis);
    node->left = buildTree(points, depth + 1, start, mid);
    node->right = buildTree(points, depth + 1, mid + 1, end);

    return node;
}

void KDTree::build(std::vector<Point> points) {
    treeSize = points.size();
    if (treeSize == 0) {
        root = nullptr;
        return;
    }
    root = buildTree(points, 0, 0, points.size());
}

void KDTree::searchKNearest(const KDNode* node, const Point& target, int k,
                            std::vector<std::pair<double, Point>>& nearest) const {
    if (!node) {
        return;
    }

    double dist = target.distanceTo(node->point);

    // Add to nearest list
    if (nearest.size() < static_cast<size_t>(k)) {
        nearest.push_back({dist, node->point});
        std::push_heap(nearest.begin(), nearest.end());
    } else if (dist < nearest.front().first) {
        std::pop_heap(nearest.begin(), nearest.end());
        nearest.back() = {dist, node->point};
        std::push_heap(nearest.begin(), nearest.end());
    }

    // Determine which side to search first
    double diff = node->axis == 0 ? target.x - node->point.x : target.y - node->point.y;
    KDNode* nearSide = diff < 0 ? node->left.get() : node->right.get();
    KDNode* farSide = diff < 0 ? node->right.get() : node->left.get();

    // Search near side
    searchKNearest(nearSide, target, k, nearest);

    // Check if we need to search far side
    if (nearest.size() < static_cast<size_t>(k) || std::abs(diff) < nearest.front().first) {
        searchKNearest(farSide, target, k, nearest);
    }
}

std::vector<std::pair<double, Point>> KDTree::kNearestNeighbors(const Point& target,
                                                                 int k) const {
    if (!root || k <= 0) {
        return {};
    }

    std::vector<std::pair<double, Point>> nearest;
    nearest.reserve(k);

    searchKNearest(root.get(), target, k, nearest);

    // Sort by distance (ascending)
    std::sort(nearest.begin(), nearest.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    return nearest;
}

void KDTree::rangeSearch(const KDNode* node, double minX, double maxX, double minY, double maxY,
                         std::vector<Point>& results) const {
    if (!node) {
        return;
    }

    // Check if current point is in range
    if (node->point.x >= minX && node->point.x <= maxX && node->point.y >= minY &&
        node->point.y <= maxY) {
        results.push_back(node->point);
    }

    // Determine which children to search
    if (node->axis == 0) {  // Split on x
        if (minX <= node->point.x) {
            rangeSearch(node->left.get(), minX, maxX, minY, maxY, results);
        }
        if (maxX >= node->point.x) {
            rangeSearch(node->right.get(), minX, maxX, minY, maxY, results);
        }
    } else {  // Split on y
        if (minY <= node->point.y) {
            rangeSearch(node->left.get(), minX, maxX, minY, maxY, results);
        }
        if (maxY >= node->point.y) {
            rangeSearch(node->right.get(), minX, maxX, minY, maxY, results);
        }
    }
}

std::vector<Point> KDTree::rangeQuery(double minX, double maxX, double minY, double maxY) const {
    std::vector<Point> results;
    if (!root) {
        return results;
    }

    rangeSearch(root.get(), minX, maxX, minY, maxY, results);
    return results;
}
