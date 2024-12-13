# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "python-dotenv",
#   "scipy",
#   "numpy",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import httpx
import chardet
import numpy as np
from scipy.stats import skew, kurtosis
from dotenv import load_dotenv

# Force non-interactive matplotlib backend
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform detailed data analysis."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict(),
        'skewness': {col: skew(df[col].dropna()) for col in numeric_df.columns},
        'kurtosis': {col: kurtosis(df[col].dropna()) for col in numeric_df.columns}
    }
    
    return analysis

def visualize_data(df, output_dir):
    """Generate and save comprehensive visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # Pairplot for numeric data (if there are few columns)
    if len(numeric_columns) <= 5:
        sns.pairplot(df[numeric_columns].dropna())
        plt.savefig(os.path.join(output_dir, 'pairplot.png'))
        plt.close()

    # Distribution plots
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column].dropna(), kde=True, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
        plt.close()

def generate_narrative(analysis):
    """Generate a detailed narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    prompt = (
        "Provide a detailed analysis based on the following data summary:\n\n"
        f"Summary Statistics:\n{analysis['summary']}\n\n"
        f"Missing Values:\n{analysis['missing_values']}\n\n"
        f"Correlation:\n{analysis['correlation']}\n\n"
        f"Skewness:\n{analysis['skewness']}\n\n"
        f"Kurtosis:\n{analysis['kurtosis']}\n\n"
        "Generate insights and potential actions based on these findings."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return "Narrative generation failed due to an error."

def save_narrative(output_dir, narrative):
    """Save the generated narrative to a README file."""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(narrative)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze datasets and generate insights.")
    parser.add_argument("file_path", help="Path to the dataset CSV file.")
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.file_path)

    # Analyze data
    analysis = analyze_data(df)

    # Visualize data
    visualize_data(df, args.output_dir)

    # Generate and save narrative
    narrative = generate_narrative(analysis)
    save_narrative(args.output_dir, narrative)

    print(f"Analysis complete. Outputs saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
