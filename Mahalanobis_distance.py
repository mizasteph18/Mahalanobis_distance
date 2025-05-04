import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

def plot_sensitivities_interactive(df, sensitivity_columns, date_column='date', 
                                   stress_test_column='stress_test',
                                   client_category=None, client_category_column=None):
    """
    Create an interactive plot with checkboxes to toggle visibility of sensitivity lines.
    
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
    client_category : str or None
        If provided, filter data to this specific category
    client_category_column : str or None
        Column name for client category if applicable
    """
    # Filter by category if specified
    if client_category is not None and client_category_column is not None:
        plot_df = df[df[client_category_column] == client_category].copy()
        title_suffix = f" - {client_category}"
    else:
        plot_df = df.copy()
        title_suffix = ""
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"Stress Test and Sensitivity Analysis{title_suffix}", fontsize=16)
    
    # Plot stress test values in top subplot
    ax1.plot(plot_df[date_column], plot_df[stress_test_column], 
             marker='o', linestyle='-', color='black', label='Stress Test')
    
    # Add outliers if available
    if 'is_outlier' in plot_df.columns:
        outliers = plot_df[plot_df['is_outlier']]
        if not outliers.empty:
            ax1.scatter(outliers[date_column], outliers[stress_test_column], 
                        color='red', s=100, zorder=5, label='Outliers')
    
    ax1.set_ylabel('Stress Test Value')
    ax1.set_title('Stress Test Values Over Time')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot all sensitivities in bottom subplot
    lines = []
    for col in sensitivity_columns:
        line, = ax2.plot(plot_df[date_column], plot_df[col], 
                        marker='.', linestyle='-', label=col)
        lines.append(line)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sensitivity Value')
    ax2.set_title('Sensitivity Measures Over Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    # Add checkboxes to toggle visibility of each sensitivity
    plt.subplots_adjust(left=0.2)
    
    # Create a frame for the checkboxes (in figure coordinates)
    rax = plt.axes([0.02, 0.4, 0.12, 0.15])
    check = CheckButtons(
        ax=rax,
        labels=sensitivity_columns,
        actives=[True] * len(sensitivity_columns)  # All sensitivities visible initially
    )
    
    # Function to toggle line visibility
    def toggle_visibility(label):
        index = sensitivity_columns.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        fig.canvas.draw_idle()
    
    # Connect the callback
    check.on_clicked(toggle_visibility)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.2)
    
    return fig, ax1, ax2

# Add simplified version for non-interactive environments
def plot_sensitivities_simple(df, sensitivity_columns, date_column='date', 
                             stress_test_column='stress_test',
                             selected_sensitivities=None,
                             client_category=None, client_category_column=None):
    """
    Plot stress test and selected sensitivities without interactive features.
    
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
    selected_sensitivities : list or None
        List of sensitivity columns to plot. If None, all sensitivities are plotted.
    client_category : str or None
        If provided, filter data to this specific category
    client_category_column : str or None
        Column name for client category if applicable
    """
    # Default to all sensitivities if none specified
    if selected_sensitivities is None:
        selected_sensitivities = sensitivity_columns
    
    # Filter by category if specified
    if client_category is not None and client_category_column is not None:
        plot_df = df[df[client_category_column] == client_category].copy()
        title_suffix = f" - {client_category}"
    else:
        plot_df = df.copy()
        title_suffix = ""
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Stress Test and Sensitivity Analysis{title_suffix}", fontsize=16)
    
    # Plot stress test values in top subplot
    ax1.plot(plot_df[date_column], plot_df[stress_test_column], 
             marker='o', linestyle='-', color='black', label='Stress Test')
    
    # Add outliers if available
    if 'is_outlier' in plot_df.columns:
        outliers = plot_df[plot_df['is_outlier']]
        if not outliers.empty:
            ax1.scatter(outliers[date_column], outliers[stress_test_column], 
                        color='red', s=100, zorder=5, label='Outliers')
    
    ax1.set_ylabel('Stress Test Value')
    ax1.set_title('Stress Test Values Over Time')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot selected sensitivities in bottom subplot
    for col in selected_sensitivities:
        ax2.plot(plot_df[date_column], plot_df[col], 
                marker='.', linestyle='-', label=col)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sensitivity Value')
    ax2.set_title(f'Selected Sensitivity Measures Over Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return fig, ax1, ax2

# Example usage
if __name__ == "__main__":
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
    
    # Create interactive plot
    print("Creating interactive plot with checkboxes to toggle sensitivities...")
    fig, _, _ = plot_sensitivities_interactive(
        df, 
        sensitivity_columns, 
        date_column='date',
        stress_test_column='stress_test'
    )
    plt.show()
    
    # Create non-interactive plot with selected sensitivities
    print("\nCreating non-interactive plot with selected sensitivities...")
    fig, _, _ = plot_sensitivities_simple(
        df, 
        sensitivity_columns, 
        date_column='date',
        stress_test_column='stress_test',
        selected_sensitivities=['sensitivity_1', 'sensitivity_3', 'sensitivity_5']  # Only show these
    )
    plt.show()
    
    # Example with client category filter
    print("\nCreating plot filtered by client category...")
    fig, _, _ = plot_sensitivities_simple(
        df, 
        sensitivity_columns, 
        date_column='date',
        stress_test_column='stress_test',
        client_category='High Risk',
        client_category_column='client_category'
    )
    plt.show()
