# Mahalanobis_distance

# Stress Test Outlier Detection Framework

## Overview

This Python package provides advanced statistical tools for detecting outliers in financial stress test data using weighted Mahalanobis distance with time decay. It is designed for risk managers, financial analysts, and regulators who need to monitor and validate stress test results against various risk sensitivity measures.

## Key Features

- **Time-Weighted Mahalanobis Distance**: Detects multivariate outliers while giving more weight to recent observations
- **Exponential Decay**: Configurable decay factor to control the influence of historical observations
- **Statistical Confidence Levels**: Threshold determination based on chi-squared distribution
- **Client Categorization**: Support for separate analysis by client risk categories
- **Comprehensive Visualization**: Multiple plotting functions for time series, distances, and sensitivity contributions
- **Interactive Mode**: Jupyter widget support for interactive analysis (when run in compatible environments)

## Technical Details

### Core Algorithm

The detection algorithm implements several advanced statistical concepts:

1. **Time-Weighted Statistics**: Calculates weighted means and covariance matrices where weights decay exponentially with time
2. **Mahalanobis Distance**: Measures the distance between a point and a distribution in multivariate space, accounting for correlations
3. **Rolling Window Analysis**: Each observation is evaluated against the distribution of prior observations
4. **Statistical Threshold**: Uses chi-squared distribution with degrees of freedom equal to the number of sensitivity variables

### Mathematical Foundation

The Mahalanobis distance is calculated as:

```
D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Where:
- x is the vector of sensitivity values for the current observation
- μ is the weighted mean vector of historical sensitivity values
- Σ is the weighted covariance matrix of historical sensitivity values

Time weights are calculated as:

```
w = exp(-decay_factor * days_from_latest)
```

### Performance Considerations

- Regularization is applied to the covariance matrix to ensure invertibility
- Matrix operations are optimized using NumPy's vectorized operations
- Time complexity is approximately O(n²) for n observations
