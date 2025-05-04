
import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display

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

def plot_sensitivities(df, sensitivity_columns, date_column='date', 
                       stress_test_column='stress_test', 
                       client_category_column=None, 
                       category=None,
                       selected_sensitivities=None):
    """
    Plot selected sensitivities along with stress test values over time.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dates, stress test figures, and sensitivity measures
    sensitivity_columns : list
        List of column names containing sensitivity measures
    date_column : str
        Column name for date
    stress_test_column : str
        Column name for stress test figures
    client_category_column : str or None
        Column name for client category if applicable
    category : str or None
        Specific category to plot if client_category_column is provided
    selected_sensitivities : list or None
        List of sensitivity columns to plot. If None, all sensitivities are plotted.
    """
    # Filter by category if specified
    if client_category_column is not None and category is not None:
        plot_df = df[df[client_category_column] == category].copy()
    else:
        plot_df = df.copy()
    
    # Determine which sensitivities to plot
    if selected_sensitivities is None:
        selected_sensitivities = sensitivity_columns
    
    # Create figure with two subplots sharing x-axis
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Title with category if provided
    title_suffix = f" for {category}" if category else ""
    fig.suptitle(f"Stress Test and Sensitivity Analysis{title_suffix}", fontsize=16)
    
    # Plot stress test values in top subplot
    axes[0].plot(plot_df[date_column], plot_df[stress_test_column], 
                marker='o', linestyle='-', color='black', label='Stress Test Value')
    
    # Add outliers if available
    if 'is_outlier' in plot_df.columns:
        outliers = plot_df[plot_df['is_outlier']]
        if not outliers.empty:
            axes[0].scatter(outliers[date_column], outliers[stress_test_column], 
                           color='red', s=100, zorder=5, label='Outliers')
    
    axes[0].set_ylabel('Stress Test Value')
    axes[0].set_title('Stress Test Values Over Time')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot selected sensitivities in bottom subplot
    for col in selected_sensitivities:
        axes[1].plot(plot_df[date_column], plot_df[col], marker='.', linestyle='-', label=col)
    
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Sensitivity Value')
    axes[1].set_title('Selected Sensitivity Measures Over Time')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def interactive_sensitivity_plot(df, sensitivity_columns, date_column='date', 
                               stress_test_column='stress_test',
                               client_category_column=None):
    """
    Create interactive widgets to select which sensitivities to plot.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dates, stress test figures, and sensitivity measures
    sensitivity_columns : list
        List of column names containing sensitivity measures
    date_column : str
        Column name for date
    stress_test_column : str
        Column name for stress test figures
    client_category_column : str or None
        Column name for client category if applicable
    """
    # Create category selection widget if applicable
    category_widget = None
    if client_category_column is not None:
        categories = sorted(df[client_category_column].unique())
        category_widget = widgets.Dropdown(
            options=categories,
            description='Client Category:',
            disabled=False
        )
    
    # Create sensitivity selection widget
    sensitivity_widget = widgets.SelectMultiple(
        options=sensitivity_columns,
        value=sensitivity_columns[:min(3, len(sensitivity_columns))],  # Default to first 3 sensitivities
        description='Sensitivities:',
        disabled=False
    )
    
    # Create update button
    update_button = widgets.Button(
        description='Update Plot',
        disabled=False,
        button_style='primary',
        tooltip='Click to update the plot with selected sensitivities'
    )
    
    # Display widgets
    if category_widget is not None:
        display(widgets.VBox([category_widget, sensitivity_widget, update_button]))
    else:
        display(widgets.VBox([sensitivity_widget, update_button]))
    
    # Create function to handle button click
    def on_button_click(b):
        selected_sensitivities = sensitivity_widget.value
        
        if not selected_sensitivities:
            print("Please select at least one sensitivity to plot.")
            return
        
        category = category_widget.value if category_widget is not None else None
        
        plot_sensitivities(
            df, 
            sensitivity_columns, 
            date_column=date_column,
            stress_test_column=stress_test_column,
            client_category_column=client_category_column,
            category=category,
            selected_sensitivities=selected_sensitivities
        )
    
    # Register the callback
    update_button.on_click(on_button_click)

def analyze_latest_stress_test(df, sensitivity_columns, stress_test_column='stress_test',
                              date_column='date', decay_factor=0.05, confidence_level=0.99,
                              client_category_column=None, show_plot=True, 
                              interactive_sensitivity_selection=False):
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
    interactive_sensitivity_selection : bool
        Whether to provide interactive widgets for selecting sensitivities to plot
        
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
    
    # Create interactive sensitivity plot if requested
    if interactive_sensitivity_selection:
        print("\nInteractive Sensitivity Selection:")
        if client_category_column is not None:
            interactive_sensitivity_plot(
                df, sensitivity_columns, date_column, stress_test_column, client_category_column
            )
        else:
            interactive_sensitivity_plot(
                df, sensitivity_columns, date_column, stress_test_column
            )
    
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

def plot_sensitivity_contributions(df, sensitivity_columns, latest_only=True, n_latest=5, 
                                  date_column='date', client_category_column=None, category=None):
    """
    Plot the contribution of each sensitivity measure to the Mahalanobis distance.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with calculated Mahalanobis distances
    sensitivity_columns : list
        List of column names containing sensitivity measures
    latest_only : bool
        If True, plot only the latest observation
    n_latest : int
        Number of latest observations to plot if latest_only is False
    date_column : str
        Column name for date
    client_category_column : str or None
        Column name for client category if applicable
    category : str or None
        Specific category to plot if client_category_column is provided
    """
    # Filter by category if specified
    if client_category_column is not None and category is not None:
        plot_df = df[df[client_category_column] == category].copy()
    else:
        plot_df = df.copy()
    
    # Select observations to analyze
    if latest_only:
        plot_indices = [len(plot_df) - 1]  # Only the last row
        title_suffix = "Latest Observation"
    else:
        plot_indices = range(max(0, len(plot_df) - n_latest), len(plot_df))
        title_suffix = f"Last {n_latest} Observations"
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For each sensitivity, plot its values for the selected observations
    bar_width = 0.8 / len(sensitivity_columns)
    
    for i, col in enumerate(sensitivity_columns):
        # Extract values for the selected observations
        values = [plot_df.iloc[idx][col] for idx in plot_indices]
        
        # Calculate positions for bars
        positions = np.arange(len(plot_indices)) + i * bar_width
        
        # Plot bars
        ax.bar(positions, values, width=bar_width, label=col)
    
    # Set x-tick labels to dates
    if latest_only:
        x_labels = [plot_df.iloc[plot_indices[0]][date_column].strftime('%Y-%m-%d')]
    else:
        x_labels = [plot_df.iloc[idx][date_column].strftime('%Y-%m-%d') for idx in plot_indices]
    
    ax.set_xticks(np.arange(len(plot_indices)) + (len(sensitivity_columns) - 1) * bar_width / 2)
    ax.set_xticklabels(x_labels, rotation=45)
    
    # Set labels and title
    category_str = f" for {category}" if category else ""
    ax.set_ylabel('Sensitivity Value')
    ax.set_title(f'Sensitivity Contributions - {title_suffix}{category_str}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_non_interactive_sensitivity_plot(df, sensitivity_columns, date_column='date', 
                                           stress_test_column='stress_test',
                                           client_category_column=None, category=None,
                                           selected_sensitivities=None):
    """
    Create a plot with selected sensitivities for non-interactive environments.
    This function is useful when ipywidgets are not available.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing dates, stress test figures, and sensitivity measures
    sensitivity_columns : list
        List of column names containing sensitivity measures
    date_column : str
        Column name for date
    stress_test_column : str
        Column name for stress test figures
    client_category_column : str or None
        Column name for client category if applicable
    category : str or None
        Specific category to plot if client_category_column is provided
    selected_sensitivities : list or None
        List of sensitivity columns to plot. If None, all sensitivities are plotted.
    """
    # Default to all sensitivities if none are specified
    if selected_sensitivities is None:
        selected_sensitivities = sensitivity_columns
    
    # Filter by category if specified
    if client_category_column is not None and category is not None:
        plot_df = df[df[client_category_column] == category].copy()
    else:
        plot_df = df.copy()
    
    # Plot the selected sensitivities
    plot_sensitivities(
        plot_df, 
        sensitivity_columns, 
        date_column=date_column,
        stress_test_column=stress_test_column,
        selected_sensitivities=selected_sensitivities
    )

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
    sensi_4 = np.random.normal(75, 15, 100) + np.cos(np.linspace(0, 3*np.pi, 100)) * 25  # Another cyclical pattern
    sensi_5 = np.random.normal(150, 25, 100) - np.linspace(0, 30, 100)  # With downward trend
    
    # Create stress test values with relationship to sensitivities plus noise
    stress_test = 0.3 * sensi_1 + 0.2 * sensi_2 + 0.15 * sensi_3 + 0.25 * sensi_4 + 0.1 * sensi_5 + np.random.normal(0, 40, 100)
    
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
        'sensitivity_3': sensi_3,
        'sensitivity_4': sensi_4,
        'sensitivity_5': sensi_5
    })
    
    # Define sensitivity columns
    sensitivity_columns = ['sensitivity_1', 'sensitivity_2', 'sensitivity_3', 'sensitivity_4', 'sensitivity_5']
    
    # Example 1: Analyze without client categories, with interactive sensitivity selection
    print("\nExample 1: Interactive Analysis without Client Categories")
    results = analyze_latest_stress_test(
        df, 
        sensitivity_columns, 
        stress_test_column='stress_test',
        date_column='date',
        decay_factor=0.03,
        confidence_level=0.99,
        client_category_column=None,
        show_plot=True,
        interactive_sensitivity_selection=True  # Enable interactive sensitivity selection
    )
    
    # Example 2: For non-interactive environments, plot specific sensitivities
    print("\nExample 2: Non-interactive Plot with Selected Sensitivities")
    create_non_interactive_sensitivity_plot(
        df,
        sensitivity_columns,
        date_column='date',
        stress_test_column='stress_test',
        selected_sensitivities=['sensitivity_1', 'sensitivity_3', 'sensitivity_5']  # Plot only these sensitivities
    )
    
    # Example 3: Plot sensitivity contributions for the latest observation
    print("\nExample 3: Sensitivity Contributions for Latest Observation")
    plot_sensitivity_contributions(
        results['all'],
        sensitivity_columns,
        latest_only=True
    )
    
    # Example 4: Plot sensitivity contributions for the last 5 observations
    print("\nExample 4: Sensitivity Contributions for Last 5 Observations")
    plot_sensitivity_contributions(
        results['all'],
        sensitivity_columns,
        latest_only=False,
        n_latest=5
    )
    
    # Example 5: Analyze with client categories
    print("\nExample 5: Analysis with Client Categories")
    results_by_category = analyze_latest_stress_test(
        df, 
        sensitivity_columns, 
        stress_test_column='stress_test',
        date_column='date',
        decay_factor=0.03,
        confidence_level=0.99,
        client_category_column='client_category',  # Analyze by client category
        show_plot=True,
        interactive_sensitivity_selection=True  # Enable interactive sensitivity selection
    )
