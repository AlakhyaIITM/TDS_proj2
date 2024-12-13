import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import argparse

# Function to clean data by converting columns to numeric or date types where applicable
def clean_data(df):
    """Convert numeric columns with non-numeric values to valid types if possible."""
    for col in df.columns:
        # Attempt to convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='ignore')
        # Attempt to parse dates
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except Exception:
            pass
    return df

# Function to analyze data
def analyze_data(df):
    numeric_df = df.select_dtypes(include=['number'])
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {},
        'skewness': {col: skew(df[col].dropna()) for col in numeric_df.columns},
        'kurtosis': {col: kurtosis(df[col].dropna()) for col in numeric_df.columns}
    }
    return analysis

# Function to visualize data
def visualize_data(df, output_dir):
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
    else:
        print("No numeric columns available for correlation heatmap.")

    # Example of additional visualization (distribution of numeric columns)
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
        plt.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize data from a CSV file.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save the visualizations.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset
    try:
        df = pd.read_csv(args.input_file)
        print(f"Data loaded successfully from {args.input_file}.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Clean the data
    df = clean_data(df)

    # Analyze the data
    analysis = analyze_data(df)
    analysis_output_path = os.path.join(args.output_dir, 'analysis.json')
    with open(analysis_output_path, 'w') as f:
        pd.json.dump(analysis, f, indent=4)
    print(f"Analysis saved to {analysis_output_path}.")

    # Visualize the data
    try:
        visualize_data(df, args.output_dir)
        print(f"Visualizations saved to {args.output_dir}.")
    except Exception as e:
        print(f"Failed to visualize data: {e}")

if __name__ == "__main__":
    main()
