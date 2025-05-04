import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def detect_stress_test_outliers(df, sensitivity_columns, stress_test_column='stress_test', 
                               date_column='date', decay_factor=0.05, confidence_level=0.99):
    """
    Detect outliers in stress test figures using Mahalanobis Distance with time decay.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dates, stress test figures, and sensitivity measures
    sensitivity_columns : list
        List of column names containing sensitivity measures
    stress_test_column : str
        Column name for stress test figures
    date_column : str
        Column name for date
    decay_factor : float
        Controls how quickly the importance of past observations decays (higher = faster decay)
    confidence_level : float
        Confidence level for outlier detection (e.g., 0.95, 0.99)
        
    Returns:
    --------
    result_df : pandas DataFrame
        DataFrame with original data plus Mahalanobis distances and outlier flags
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_values(by=date_column).reset_index(drop=True)
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Calculate time differences in days from the latest date
    latest_date = df[date_column].max()
    df['days_from_latest'] = (latest_date - df[date_column]).dt.days
    
    # Calculate time weights with exponential decay
    df['time_weight'] = np.exp(-decay_factor * df['days_from_latest'])
    
    # Normalize weights to sum to 1
    df['time_weight'] = df['time_weight'] / df['time_weight'].sum()
    
    # Initialize columns for Mahalanobis distances
    df['mahalanobis_distance'] = np.nan
    
    # For each row, calculate Mahalanobis distance using data up to that point
    for i in range(len(df)):
        if i < len(sensitivity_columns) + 1:  # Need at least n+1 observations for n variables
            continue
            
        # Get historical data up to current row
        hist_data = df.iloc[:i+1].copy()
        
        # Extract current row sensitivities
        current_sensitivities = hist_data.iloc[-1][sensitivity_columns].values.reshape(1, -1)
        
        # Calculate weighted mean vector (excluding current observation)
        weights_excluding_current = hist_data.iloc[:-1]['time_weight'].values
        weights_excluding_current = weights_excluding_current / weights_excluding_current.sum()
        
        weighted_mean = np.average(
            hist_data.iloc[:-1][sensitivity_columns].values, 
            axis=0, 
            weights=weights_excluding_current
        )
        
        # Calculate weighted covariance matrix (excluding current observation)
        # Using numpy's cov with weights
        X = hist_data.iloc[:-1][sensitivity_columns].values
        w = weights_excluding_current.reshape(-1, 1)  # Reshape for broadcasting
        X_centered = X - weighted_mean
        weighted_cov = np.dot(X_centered.T * w.T, X_centered) / (1 - (w**2).sum())
        
        # Add small regularization to ensure invertibility
        weighted_cov += np.eye(len(sensitivity_columns)) * 1e-6
        
        try:
            # Calculate inverse of covariance matrix
            inv_cov = np.linalg.inv(weighted_cov)
            
            # Calculate Mahalanobis distance
            diff = current_sensitivities - weighted_mean
            mahalanobis_sq = np.dot(np.dot(diff, inv_cov), diff.T)[0, 0]
            df.loc[i, 'mahalanobis_distance'] = np.sqrt(mahalanobis_sq)
        except np.linalg.LinAlgError:
            # If matrix inversion fails, set distance to NaN
            df.loc[i, 'mahalanobis_distance'] = np.nan
    
    # Calculate threshold based on chi-squared distribution
    dof = len(sensitivity_columns)  # degrees of freedom = number of variables
    threshold = np.sqrt(chi2.ppf(confidence_level, dof))
    
    # Flag outliers
    df['is_outlier'] = df['mahalanobis_distance'] > threshold
    
    # Add calculated threshold for reference
    df['threshold'] = threshold
    
    return df

def analyze_latest_stress_test(df, sensitivity_columns, stress_test_column='stress_test',
                              date_column='date', decay_factor=0.05, confidence_level=0.99,
                              client_category_column=None, show_plot=True):
    """
    Analyze if the latest stress test figure is an outlier, with optional grouping by client category.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dates, stress test figures, and sensitivity measures
    sensitivity_columns : list
        List of column names containing sensitivity measures
    stress_test_column : str
        Column name for stress test figures
    date_column : str
        Column name for date
    decay_factor : float
        Controls how quickly the importance of past observations decays (higher = faster decay)
    confidence_level : float
        Confidence level for outlier detection (e.g., 0.95, 0.99)
    client_category_column : str or None
        If provided, analysis will be done separately for each client category
    show_plot : bool
        Whether to display plots of the results
        
    Returns:
    --------
    dict : Dictionary of results for each client category (or 'all' if no category provided)
    """
    results = {}
    
    if client_category_column is not None:
        # Analyze separately for each client category
        for category, group_df in df.groupby(client_category_column):
            print(f"\nAnalyzing client category: {category}")
            result_df = detect_stress_test_outliers(
                group_df, sensitivity_columns, stress_test_column, 
                date_column, decay_factor, confidence_level
            )
            
            # Store results
            results[category] = result_df
            
            # Check if latest observation is an outlier
            latest_row = result_df.iloc[-1]
            is_outlier = latest_row['is_outlier']
            distance = latest_row['mahalanobis_distance']
            threshold = latest_row['threshold']
            
            print(f"Latest stress test figure ({latest_row[stress_test_column]}) for {category}:")
            print(f"Date: {latest_row[date_column]}")
            print(f"Mahalanobis distance: {distance:.4f}")
            print(f"Threshold at {confidence_level*100}% confidence: {threshold:.4f}")
            print(f"Outlier status: {'OUTLIER' if is_outlier else 'Normal'}")
            
            if show_plot:
                plot_results(result_df, date_column, stress_test_column, category)
    else:
        # Analyze entire dataset
        result_df = detect_stress_test_outliers(
            df, sensitivity_columns, stress_test_column, 
            date_column, decay_factor, confidence_level
        )
        
        # Store results
        results['all'] = result_df
        
        # Check if latest observation is an outlier
        latest_row = result_df.iloc[-1]
        is_outlier = latest_row['is_outlier']
        distance = latest_row['mahalanobis_distance']
        threshold = latest_row['threshold']
        
        print(f"\nLatest stress test figure ({latest_row[stress_test_column]}):")
        print(f"Date: {latest_row[date_column]}")
        print(f"Mahalanobis distance: {distance:.4f}")
        print(f"Threshold at {confidence_level*100}% confidence: {threshold:.4f}")
        print(f"Outlier status: {'OUTLIER' if is_outlier else 'Normal'}")
        
        if show_plot:
            plot_results(result_df, date_column, stress_test_column)
    
    return results

def plot_results(df, date_column, stress_test_column, category=None):
    """
    Plot the results of the outlier detection.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with calculated Mahalanobis distances and outlier flags
    date_column : str
        Column name for date
    stress_test_column : str
        Column name for stress test figures
    category : str or None
        Client category if applicable
    """
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Title with category if provided
    title_suffix = f" for {category}" if category else ""
    fig.suptitle(f"Stress Test Outlier Analysis{title_suffix}", fontsize=16)
    
    # Plot 1: Stress test values over time
    axes[0].plot(df[date_column], df[stress_test_column], marker='o', linestyle='-', label='Stress Test Value')
    
    # Highlight outliers in red
    outliers = df[df['is_outlier']]
    if not outliers.empty:
        axes[0].scatter(outliers[date_column], outliers[stress_test_column], 
                       color='red', s=100, zorder=5, label='Outliers')
    
    axes[0].set_ylabel('Stress Test Value')
    axes[0].set_title('Stress Test Values Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Mahalanobis distances over time with threshold
    axes[1].plot(df[date_column], df['mahalanobis_distance'], marker='o', linestyle='-', label='Mahalanobis Distance')
    axes[1].axhline(y=df['threshold'].iloc[-1], color='r', linestyle='--', label=f"Threshold ({df['threshold'].iloc[-1]:.2f})")
    
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Mahalanobis Distance')
    axes[1].set_title('Mahalanobis Distances Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # This is a synthetic example - replace with your actual data
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create multiple sensitivity measures
    sensi_1 = np.random.normal(100, 20, 100) + np.linspace(0, 50, 100)  # With upward trend
    sensi_2 = np.random.normal(50, 10, 100) + np.sin(np.linspace(0, 4*np.pi, 100)) * 20  # With cyclical pattern
    sensi_3 = np.random.normal(200, 30, 100)  # Random fluctuation
    
    # Create stress test values with relationship to sensitivities plus noise
    stress_test = 0.5 * sensi_1 + 0.3 * sensi_2 + 0.2 * sensi_3 + np.random.normal(0, 50, 100)
    
    # Introduce an anomaly in the last observation
    stress_test[-1] = stress_test[-1] * 1.5  # Increase the last value by 50%
    
    # Create two client categories
    categories = np.random.choice(['High Risk', 'Low Risk'], 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'client_category': categories,
        'stress_test': stress_test,
        'sensitivity_1': sensi_1,
        'sensitivity_2': sensi_2,
        'sensitivity_3': sensi_3
    })
    
    # Define sensitivity columns
    sensitivity_columns = ['sensitivity_1', 'sensitivity_2', 'sensitivity_3']
    
    # Analyze without client categories
    results = analyze_latest_stress_test(
        df, 
        sensitivity_columns, 
        stress_test_column='stress_test',
        date_column='date',
        decay_factor=0.03,  # Lower value gives slower decay (more weight to history)
        confidence_level=0.99,
        client_category_column=None,  # Remove None to analyze by client category
        show_plot=True
    )
    
    # Access the results for further analysis if needed
    # full_results = results['all']  # When not using client categories
    # or when using client categories:
    # high_risk_results = results['High Risk']
    # low_risk_results = results['Low Risk']
