#include <dataset.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>

namespace Dataset {

std::vector<Point> generateClusters(const std::vector<ClusterConfig>& configs,
                                    unsigned int seed) {
    std::vector<Point> dataset;

    // Use provided seed or generate random one
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : seed);

    for (const auto& config : configs) {
        std::normal_distribution<double> distX(config.centerX, config.stdDev);
        std::normal_distribution<double> distY(config.centerY, config.stdDev);

        for (int i = 0; i < config.numPoints; ++i) {
            double x = distX(gen);
            double y = distY(gen);
            dataset.emplace_back(x, y, config.label);
        }
    }

    return dataset;
}

std::vector<Point> generateSimpleDataset(int numPointsPerClass, unsigned int seed) {
    std::vector<ClusterConfig> configs = {
        ClusterConfig(5.0, 5.0, 1.5, 0, numPointsPerClass),    // Class 0: center at (5, 5)
        ClusterConfig(15.0, 15.0, 1.5, 1, numPointsPerClass)  // Class 1: center at (15, 15)
    };

    return generateClusters(configs, seed);
}

std::pair<std::vector<Point>, std::vector<Point>> trainTestSplit(const std::vector<Point>& data,
                                                                   double trainRatio,
                                                                   bool shuffle,
                                                                   unsigned int seed) {
    if (trainRatio < 0.0 || trainRatio > 1.0) {
        throw std::invalid_argument("trainRatio must be between 0.0 and 1.0");
    }

    std::vector<Point> dataCopy = data;

    if (shuffle) {
        std::random_device rd;
        std::mt19937 gen(seed == 0 ? rd() : seed);
        std::shuffle(dataCopy.begin(), dataCopy.end(), gen);
    }

    size_t trainSize = static_cast<size_t>(dataCopy.size() * trainRatio);

    std::vector<Point> train(dataCopy.begin(), dataCopy.begin() + trainSize);
    std::vector<Point> test(dataCopy.begin() + trainSize, dataCopy.end());

    return {train, test};
}

double calculateAccuracy(const std::vector<int>& predictions,
                         const std::vector<int>& groundTruth) {
    if (predictions.size() != groundTruth.size()) {
        throw std::invalid_argument("predictions and groundTruth must have same size");
    }

    if (predictions.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == groundTruth[i]) {
            ++correct;
        }
    }

    return (static_cast<double>(correct) / predictions.size()) * 100.0;
}

std::string getDatasetStats(const std::vector<Point>& data) {
    if (data.empty()) {
        return "Empty dataset";
    }

    // Count labels
    std::map<int, int> labelCounts;
    for (const auto& point : data) {
        labelCounts[point.label]++;
    }

    // Calculate bounds
    auto [minX, maxX, minY, maxY] = getBounds(data);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Dataset Statistics:\n";
    oss << "  Total points: " << data.size() << "\n";
    oss << "  Number of classes: " << labelCounts.size() << "\n";
    oss << "  Class distribution:\n";
    for (const auto& [label, count] : labelCounts) {
        double percentage = (static_cast<double>(count) / data.size()) * 100.0;
        oss << "    Class " << label << ": " << count << " points (" << percentage << "%)\n";
    }
    oss << "  Bounds: X[" << minX << ", " << maxX << "], Y[" << minY << ", " << maxY << "]";

    return oss.str();
}

std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& predictions,
                                               const std::vector<int>& groundTruth,
                                               int numClasses) {
    if (predictions.size() != groundTruth.size()) {
        throw std::invalid_argument("predictions and groundTruth must have same size");
    }

    std::vector<std::vector<int>> matrix(numClasses, std::vector<int>(numClasses, 0));

    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions[i];
        int actual = groundTruth[i];
        if (pred >= 0 && pred < numClasses && actual >= 0 && actual < numClasses) {
            matrix[actual][pred]++;
        }
    }

    return matrix;
}

bool saveToCSV(const std::vector<Point>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "x,y,label\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& point : data) {
        file << point.x << "," << point.y << "," << point.label << "\n";
    }

    file.close();
    return true;
}

std::vector<Point> loadFromCSV(const std::string& filename) {
    std::vector<Point> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        return data;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string xStr, yStr, labelStr;

        if (std::getline(iss, xStr, ',') && std::getline(iss, yStr, ',') &&
            std::getline(iss, labelStr, ',')) {
            try {
                double x = std::stod(xStr);
                double y = std::stod(yStr);
                int label = std::stoi(labelStr);
                data.emplace_back(x, y, label);
            } catch (...) {
                // Skip malformed lines
                continue;
            }
        }
    }

    file.close();
    return data;
}

void normalize(std::vector<Point>& data) {
    if (data.empty()) {
        return;
    }

    auto [minX, maxX, minY, maxY] = getBounds(data);

    double rangeX = maxX - minX;
    double rangeY = maxY - minY;

    if (rangeX == 0.0) rangeX = 1.0;
    if (rangeY == 0.0) rangeY = 1.0;

    for (auto& point : data) {
        point.x = (point.x - minX) / rangeX;
        point.y = (point.y - minY) / rangeY;
    }
}

std::tuple<double, double, double, double> getBounds(const std::vector<Point>& data) {
    if (data.empty()) {
        return {0.0, 0.0, 0.0, 0.0};
    }

    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : data) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    return {minX, maxX, minY, maxY};
}

}  // namespace Dataset
