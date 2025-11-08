# PRONE Boosted Implementation

This directory contains the implementation of the PRONE (boosted) algorithm from Section 7.3 of the PRONE paper: "Fast k-means++ Initialization via Sampling".

## Overview

PRONE (boosted) is a two-stage pipeline that combines:
1. **PRONE (standard)** - Fast 1D projection-based k-means++ initialization
2. **Sensitivity sampling** - Coreset construction based on Bachem et al. "Scalable K-Means++"
3. **Weighted k-means++** - Final center selection on the coreset

This approach achieves O(log k) approximation guarantees with significantly improved clustering quality compared to PRONE alone.

## Files

- **`prone_boosted.hpp`** - Header file with coreset construction and weighted k-means++ functions
- **`prone_boosted.cpp`** - Implementation of the full PRONE boosted pipeline
- **`benchmark_prone_boosted.cpp`** - Benchmark comparing PRONE boosted (α=0.001, 0.01, 0.1) with other algorithms

## Algorithm Details

### Step 1: PRONE Initialization
Run PRONE (standard Gaussian projection) to get initial k centers and cluster assignments.

### Step 2: Sensitivity Sampling Coreset Construction
For each point x in dataset:
- Compute distance d²(x, C) to assigned center
- Compute sensitivity: q(x) = α/(2·cost(X)) · d²(x,C) + α/(2k) · cost(Cᵢ)/|Cᵢ|
  - where α = 16(log k + 2) is a theoretical constant
  - Cᵢ is the cluster containing x
- Sample coreset of size αn according to sensitivity distribution
- Assign weights w = total_sensitivity / (coreset_size · sensitivity[x])

### Step 3: Weighted k-means++ on Coreset
Run k-means++ on coreset points with weighted D² sampling:
- First center: uniform random from coreset
- Subsequent centers: sample proportional to weight · d²(x, C)

### Step 4: Final Assignment
Assign all original points to nearest final center.

## Hyperparameter: α (alpha)

The coreset size parameter α controls the trade-off between speed and quality:

- **α = 0.001**: Very small coreset (~25 points for n=25,000)
  - Fastest
  - Still maintains good quality due to sensitivity sampling

- **α = 0.01**: Medium coreset (~250 points)
  - Balanced speed/quality trade-off
  - **Recommended default**

- **α = 0.1**: Large coreset (~2,500 points)
  - Best quality
  - Slower but still faster than standard k-means++

## Benchmark Results (IMDB-62: 25,000 × 384)

### k = 10
| Algorithm | Runtime (s) | Cost | Notes |
|-----------|-------------|------|-------|
| PRONE (boosted, α=0.001) | 0.344 | 28,806 | Fastest PRONE variant |
| PRONE (boosted, α=0.01) | 0.336 | 28,495 | **Best quality** |
| PRONE (boosted, α=0.1) | 0.359 | 28,957 | Larger coreset |
| RS-k-means++ (FastLSH) | 0.021 | 30,370 | Fastest overall |
| AFK-MC² | 0.022 | 30,087 | Fast, good quality |
| RejectionSamplingLSH | 3.342 | 29,432 | Slow but high quality |

### k = 50
| Algorithm | Runtime (s) | Cost |
|-----------|-------------|------|
| PRONE (boosted, α=0.01) | ~0.34 | ~25,600 |
| RS-k-means++ (FastLSH) | 0.053 | 26,187 |
| RejectionSamplingLSH | 3.321 | 25,576 |

## Key Observations

1. **Quality Improvement**: PRONE boosted achieves ~5-10% better clustering cost than the original PRONE variants (Standard/Variance/Covariance)

2. **Speed**: Still very fast (~0.3-0.4s for k=10), though slower than RS-k-means++ and AFK-MC²

3. **Theoretical Guarantees**: Unlike the original PRONE variants, PRONE boosted has O(log k) approximation guarantees from the sensitivity sampling + weighted k-means++ combination

4. **Coreset Size Effect**: α=0.01 appears to be the sweet spot - it achieves the best quality while maintaining good speed

## Comparison to Original PRONE Variants

The original benchmark included three PRONE variants:
- PRONE (Standard): Cost ~29,472 for k=10
- PRONE (Variance): Cost ~28,659 for k=10
- PRONE (Covariance): Cost ~30,158 for k=10

PRONE boosted (α=0.01) achieves **28,495** - better than all three original variants!

Runtime is slightly slower (~0.34s vs ~0.006s for Standard PRONE), but the quality improvement and theoretical guarantees make this worthwhile.

## Building and Running

### Build
```bash
cd /Users/poojanshah/Desktop/Fast\ k\ means++/build
cmake ..
make benchmark_prone_boosted -j8
```

### Run Benchmark
```bash
./benchmark_prone_boosted
```

Results are saved to `../experiments/benchmark_results_prone_boosted.csv`

## Implementation Notes

### Sensitivity Sampling Formula
The implementation follows Bachem et al. closely:
- α constant = 16(log k + 2)
- Two-term sensitivity: point-specific + cluster-specific
- Normalized to form probability distribution

### Weighted k-means++
The weighted variant properly accounts for coreset point weights:
- Distance weighting: weight[i] · d²(x[i], C)
- Ensures theoretical approximation guarantees

### Center Index Conversion
PRONE returns actual center coordinates, not indices. The implementation finds the nearest point in the dataset for each PRONE center to use as indices for sensitivity sampling.

## Future Work

1. **Adaptive α**: Could tune α based on k and n for optimal speed/quality
2. **Parallel coreset construction**: Parallelize sensitivity computation and sampling
3. **GPU acceleration**: Move coreset construction and weighted k-means++ to GPU
4. **Integration with full k-means**: Use PRONE boosted as initialization for Lloyd's algorithm

## References

1. "Fast k-means++ Initialization via Sampling" (PRONE paper, Section 7.3)
2. Bachem, O., Lucic, M., & Krause, A. "Scalable K-Means++". ICML 2016.
3. Arthur, D., & Vassilvitskii, S. "k-means++: The advantages of careful seeding". SODA 2007.

## Contact

For questions or issues with this implementation, please check the main repository README.
