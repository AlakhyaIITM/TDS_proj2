# PCA and Data Analysis Script

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Function for PCA using NumPy
def pca_numpy(data, num_components):
    """Perform PCA using NumPy."""
    # Standardizing data
    mean = np.mean(data, axis=0)
    data_std = data - mean

    # Covariance matrix
    covariance_matrix = np.cov(data_std.T)

    # Eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Sorting eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    # Selecting top components
    principal_components = np.dot(data_std, eigen_vectors[:, :num_components])
    return principal_components, eigen_values[:num_components]

# Function to generate correlation heatmap
def generate_correlation_heatmap(df, output_file):
    """Generate and save a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(output_file)
    plt.close()

# Function to handle multiple datasets
def analyze_dataset(file_path, output_prefix):
    """Analyze a dataset, perform PCA, and generate visualizations."""
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        # Select numeric data
        numeric_data = df.select_dtypes(include=[np.number])

        # Perform PCA
        principal_components, explained_variance = pca_numpy(numeric_data, num_components=2)

        # Save PCA results
        pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        pca_df.to_csv(f"{output_prefix}_pca_results.csv", index=False)

        # Generate heatmap
        generate_correlation_heatmap(numeric_data, f"{output_prefix}_heatmap.png")

        print(f"Analysis completed for {file_path}. Results saved with prefix {output_prefix}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Parallel processing for multiple datasets
def process_datasets_parallel(datasets):
    """Process multiple datasets in parallel."""
    with ThreadPoolExecutor() as executor:
        for dataset, prefix in datasets:
            executor.submit(analyze_dataset, dataset, prefix)

if __name__ == "__main__":
    # Input files and output prefixes
    datasets = [
        ("goodreads.csv", "goodreads"),
        ("happiness.csv", "happiness"),
        ("media.csv", "media")
    ]

    # Process datasets
    process_datasets_parallel(datasets)

    print("All datasets processed.")
