import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(file_path, target_col='Likes of Sample Quote'):
    """
    Loads data, prepares features and target for a specific quote type,
    and handles missing values.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    # Determine the metric prefix based on the target column
    if 'Sample' in target_col:
        prefix = 'sample_'
    elif 'Popular' in target_col:
        prefix = 'popular_'
    else:
        print(f"Cannot determine metric prefix for target column: {target_col}")
        return None, None
    
    # Define features (X) and target (y)
    feature_cols = [col for col in df.columns if col.startswith(prefix)]
    numeric_feature_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    
    X = df[numeric_feature_cols]
    y = df[target_col]

    # Handle missing values by imputing with the median
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Missing values in '{col}' filled with median ({median_val:.2f}).")
            
    print(f"\nData prepared for target: '{target_col}'")
    print(f"Features ({len(X.columns)}): {', '.join(X.columns)}")
    print(f"Target: {y.name}")
    
    return X, y

def perform_shap_analysis(X, y, output_dir):
    """
    Trains a model, performs SHAP analysis, and saves plots.
    """
    print("\n--- Starting SHAP Analysis ---")
    
    # Train a LightGBM model
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X, y)
    print("Model training complete.")

    # Create a SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    print("SHAP values calculated.")

    # Create cleaner names for plotting
    clean_names = {col: col.replace('sample_', '').replace('_', ' ').title() for col in X.columns}
    X.columns = X.columns.map(clean_names)
    
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # Generate and save SHAP summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary for Predicting Quote Likes', size=14)
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to '{summary_plot_path}'")
    
    # Generate and save SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f'Feature Importance for Predicting Quote Likes', size=14)
    plt.tight_layout()
    bar_plot_path = os.path.join(output_dir, 'shap_feature_importance_bar_plot.png')
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"SHAP feature importance bar plot saved to '{bar_plot_path}'")

    print("\n--- SHAP Analysis Complete ---")


if __name__ == "__main__":
    csv_file_path = "data/outputs/master_metrics/ALOFT_master_metrics.csv"
    shap_output_path = "data/outputs/master_metrics/shap_analysis/"
    
    # We will focus on the 'Sample' quotes as they have more variance in likes
    X, y = load_and_prepare_data(csv_file_path, target_col='Likes of Sample Quote')

    if X is not None and y is not None:
        perform_shap_analysis(X, y, shap_output_path) 