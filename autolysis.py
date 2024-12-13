import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os

# Function for PCA using NumPy
def pca_numpy(data, num_components):
    """Perform PCA using NumPy."""
    mean = np.mean(data, axis=0)
    data_std = data - mean
    covariance_matrix = np.cov(data_std.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
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

# Function to generate README.md
def create_readme(output_prefix, explained_variance, heatmap_path):
    """Generate a README.md file with narrative and insights."""
    readme_content = f"""# Analysis Report for {output_prefix}

## Data Analysis Summary
This analysis involves Principal Component Analysis (PCA) and a correlation heatmap for the `{output_prefix}` dataset.

### PCA Results
The top 2 principal components explain the following variance:
- **PC1**: {explained_variance[0]:.2f}
- **PC2**: {explained_variance[1]:.2f}

### Correlation Heatmap
The correlation heatmap reveals relationships between numerical features in the dataset.

![Correlation Heatmap]({heatmap_path})

## Insights
- PCA shows that the dataset's variance can be summarized effectively in two dimensions.
- The correlation heatmap highlights the strongest relationships between variables.

## Implications
These results can be used to:
- Reduce dimensionality for further modeling.
- Identify key drivers of variance in the dataset.
    """
    with open(f"{output_prefix}/README.md", "w") as f:
        f.write(readme_content)

# Function to analyze dataset
def analyze_dataset(file_path, output_prefix):
    """Analyze a dataset, perform PCA, and generate visualizations and README."""
    try:
        os.makedirs(output_prefix, exist_ok=True)

        df = pd.read_csv(file_path)
        numeric_data = df.select_dtypes(include=[np.number])

        # Perform PCA
        principal_components, explained_variance = pca_numpy(numeric_data, num_components=2)
        pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        pca_df.to_csv(f"{output_prefix}/pca_results.csv", index=False)

        # Generate heatmap
        heatmap_path = f"{output_prefix}/heatmap.png"
        generate_correlation_heatmap(numeric_data, heatmap_path)

        # Generate README
        create_readme(output_prefix, explained_variance, heatmap_path)
        print(f"Analysis completed for {file_path}. Results saved in {output_prefix}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Parallel processing
def process_datasets_parallel(datasets):
    """Process multiple datasets in parallel."""
    with ThreadPoolExecutor() as executor:
        for dataset, prefix in datasets:
            executor.submit(analyze_dataset, dataset, prefix)

if __name__ == "__main__":
    datasets = [
        ("goodreads.csv", "goodreads"),
        ("happiness.csv", "happiness"),
        ("media.csv", "media")
    ]
    process_datasets_parallel(datasets)
    print("All datasets processed.")
