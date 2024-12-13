import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the data by converting numeric columns and handling non-numeric data."""
    for col in df.columns:
        try:
            # Convert columns to numeric where possible
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col}: {e}")
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values
    df.dropna(axis=0, how='any', inplace=True)  # Drop rows with any NaN values
    return df

def analyze_data(df):
    """Perform basic analysis and generate a summary."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "description": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return summary

def visualize_data(df, output_dir):
    """Generate visualizations including a heatmap of correlations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved correlation heatmap to {heatmap_path}")

    # Pairplot for initial exploration
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    sns.pairplot(df)
    plt.savefig(pairplot_path)
    plt.close()
    print(f"Saved pairplot to {pairplot_path}")

def save_analysis(summary, output_file):
    """Save analysis summary to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved analysis summary to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Data Analysis and Visualization Tool")
    parser.add_argument("file", help="Path to the CSV file")
    parser.add_argument("--output_dir", default="output", help="Directory to save visualizations and results")
    args = parser.parse_args()

    # Load and clean the data
    df = load_data(args.file)
    if df is None:
        return

    df = clean_data(df)
    if df.empty:
        print("The cleaned DataFrame is empty. Exiting.")
        return

    # Perform analysis
    analysis_summary = analyze_data(df)

    # Save analysis summary
    output_file = os.path.join(args.output_dir, "analysis.json")
    save_analysis(analysis_summary, output_file)

    # Generate visualizations
    visualize_data(df, args.output_dir)

if __name__ == "__main__":
    main()
