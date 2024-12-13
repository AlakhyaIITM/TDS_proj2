import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze and visualize a dataset.")
    parser.add_argument("input_file", help="Path to the CSV input file.")
    parser.add_argument("output_dir", help="Directory to save output results.")
    return parser.parse_args()

def clean_data(df):
    """Convert columns to appropriate types (numeric or date) where possible."""
    for col in df.columns:
        # Attempt to convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='ignore')
        # Attempt to parse dates
        df[col] = pd.to_datetime(df[col], errors='ignore')
    return df

def analyze_data(df):
    """Analyze the dataset and return summary statistics, correlations, skewness, and kurtosis."""
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {},
        'skewness': {col: skew(df[col].dropna()) for col in numeric_df.columns},
        'kurtosis': {col: kurtosis(df[col].dropna()) for col in numeric_df.columns}
    }
    return analysis

def visualize_data(df, output_dir):
    """Generate visualizations for the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
    else:
        print("No numeric columns available for correlation heatmap.")

    # Plot distribution for each numeric column
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
        plt.close()

def main():
    """Main function to run the analysis and visualization."""
    args = parse_arguments()
    df = pd.read_csv(args.input_file)

    # Clean the data
    df = clean_data(df)

    # Analyze the data
    analysis_results = analyze_data(df)
    print("Analysis Results:")
    print(analysis_results)

    # Save analysis results to a JSON file
    analysis_output_path = os.path.join(args.output_dir, "analysis_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame([analysis_results]).to_json(analysis_output_path, indent=4)
    print(f"Analysis results saved to {analysis_output_path}")

    # Generate visualizations
    visualize_data(df, args.output_dir)

if __name__ == "__main__":
    main()
