import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# Replace PCA with a custom implementation
class SimplePCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components_ = eigenvectors[:, :self.n_components]
        return np.dot(X_centered, self.components_)

def analyze_data(file_path):
    """Perform analysis on the dataset."""
    try:
        # Load the data
        data = pd.read_csv(file_path)
        
        # Basic stats
        print("Dataset head:")
        print(data.head())

        # Missing value check
        print("\nMissing values:")
        print(data.isnull().sum())

        # Perform a simple PCA
        numerical_data = data.select_dtypes(include=[np.number]).dropna()
        pca = SimplePCA(n_components=2)
        pca_result = pca.fit_transform(numerical_data)

        # Add PCA results to the dataframe
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]

        # Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("heatmap.png")
        plt.close()

        print("Heatmap saved as heatmap.png")

        # Save PCA results
        data.to_csv("processed_data.csv", index=False)
        print("Processed data saved as processed_data.csv")

        # Create a README
        create_readme(file_path, data)

    except Exception as e:
        print(f"Error during analysis: {e}")

def create_readme(file_path, data):
    """Generate a README file for the dataset."""
    readme_content = f"""# Dataset Analysis

## Overview

- **File Analyzed**: {os.path.basename(file_path)}
- **Number of Rows**: {data.shape[0]}
- **Number of Columns**: {data.shape[1]}

## Steps Performed
1. Checked for missing values.
2. Applied a simple PCA to reduce dimensions to 2.
3. Generated a correlation heatmap (saved as `heatmap.png`).
4. Saved the processed data as `processed_data.csv`.

## Results
- PCA components added: `PCA1`, `PCA2`
- Heatmap saved to illustrate feature correlations.

"""

    with open("README.md", "w") as f:
        f.write(readme_content)
    print("README.md file created.")

def process_files_concurrently(file_paths):
    """Process multiple files concurrently."""
    with ThreadPoolExecutor() as executor:
        executor.map(analyze_data, file_paths)

if __name__ == "__main__":
    # Example usage
    csv_files = ["media.csv"]  # Replace with your actual CSV file paths
    process_files_concurrently(csv_files)
