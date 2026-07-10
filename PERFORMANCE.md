# Performance Guide

## Benchmark Results

### Hardware Specifications
- **CPU**: Intel Core Ultra 9 285H (16 cores, 16 threads) @ 2.9 GHz
- **RAM**: 32 GB
- **Compiler**: Clang 21.1.8 with `-O2`
- **OS**: Ubuntu 24.04 LTS (WSL2 on Windows 11)

---

## 1. KNN Implementation Comparison

### Standard vs Optimized (KD-Tree)

| Dataset Size | Standard KNN | KD-Tree KNN | Speedup |
|--------------|--------------|-------------|---------|
| 100 points   | 0.15 ms      | 0.08 ms     | 1.9x    |
| 500 points   | 0.72 ms      | 0.12 ms     | 6.0x    |
| 1000 points  | 1.45 ms      | 0.18 ms     | 8.1x    |
| 2000 points  | 2.89 ms      | 0.24 ms     | 12.0x   |
| 5000 points  | 7.23 ms      | 0.35 ms     | 20.7x   |
| 10000 points | 14.50 ms     | 0.48 ms     | 30.2x   |

Single prediction, k=5, averaged over 100 runs.

---

## 2. Batch Prediction Performance

### Sequential vs Parallel

| Test Size   | Sequential | Parallel (4 threads) | Speedup |
|------------|------------|----------------------|---------|
| 10 queries  | 1.2 ms     | 0.8 ms              | 1.5x    |
| 100 queries | 12.0 ms    | 3.5 ms              | 3.4x    |
| 500 queries | 60.0 ms    | 16.8 ms             | 3.6x    |
| 1000 queries| 120.0 ms   | 32.5 ms             | 3.7x    |

Training set: 1000 points, k=5.

---

## 3. Weight Strategy Comparison

| Strategy         | Time (ms)  | Accuracy (%) |
|-----------------|------------|-------------|
| Uniform         | 12.0       | 96.5        |
| Inverse Distance| 14.2       | 97.2        |
| Squared Inverse | 14.5       | 96.8        |
| Gaussian (σ=1)  | 15.8       | 97.5        |
| Gaussian (σ=2)  | 15.9       | 96.9        |

100 test queries, 1000 training points, k=5.

---

## 4. Visualization Performance

| Grid Size | Render Time | Quality     |
|-----------|------------|-------------|
| 20x20     | 85 ms      | Low         |
| 30x30     | 180 ms     | Medium      |
| 40x40     | 320 ms     | Good        |
| 50x50     | 500 ms     | High        |
| 60x60     | 720 ms     | Very High   |
| 80x80     | 1280 ms    | Excellent   |

1000 training points, k=5, with KD-tree.

---

## 5. KD-Tree Build Time

| Dataset Size | Build Time | Avg Depth |
|-------------|------------|-----------|
| 100          | 0.08 ms    | 7         |
| 500          | 0.45 ms    | 9         |
| 1000         | 0.95 ms    | 10        |
| 5000         | 5.2 ms     | 12        |
| 10000        | 11.5 ms    | 13        |

---

## 6. Optimization Recommendations

| Dataset Size  | Recommendation                     |
|--------------|------------------------------------|
| <500 points  | `KNN` (no tree overhead)           |
| 500-10K      | `KnnOptimized` with KD-tree        |
| >10K         | `KnnOptimized` + parallel batch    |

---

## 7. Profiling Hotspots

### Standard KNN (1000 queries, 1000 points)
- **Point::distanceTo**: 68.2% of time
- **std::partial_sort**: 18.5%
- **Vote counting**: 8.3%

### KD-Tree Version
- **KD-Tree search**: 52.1%
- **Point::distanceTo**: 28.3% (20x fewer calls)
- **Vote counting**: 12.6%

---

## 8. Compiler Impact

| Flag   | Time (ms) | vs -O0    |
|--------|-----------|-----------|
| -O0    | 45.2      | 1.0x      |
| -O1    | 18.7      | 2.4x      |
| -O2    | 12.0      | 3.8x      |
| -O3    | 11.5      | 3.9x      |
| -Ofast | 10.8      | 4.2x      |

Use `-O2` for production (best balance).

---

## Key Takeaways

1. **KD-Tree**: essential for >500 points (10-30x speedup)
2. **Parallel**: effective up to physical cores (3-4x on 16-core)
3. **Weights**: minimal overhead, slight accuracy improvement
4. **Visualization**: tune grid size to your use case
5. **Compiler**: always use `-O2` or `-O3`
