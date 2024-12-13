import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform PCA using NumPy
def pca_numpy(data, n_components):
    """
    Perform Principal Component Analysis (PCA) using NumPy.

    Args:
        data (numpy.ndarray): Input data matrix of shape (samples, features).
        n_components (int): Number of principal components to retain.

    Returns:
        numpy.ndarray: Data transformed to the new principal components.
    """
    # Center the data by subtracting the mean
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_components]

    # Transform the data
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    return transformed_data

# Load dataset
def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Generate correlation heatmap
def generate_correlation_heatmap(data, output_path):
    """
    Generate a heatmap of correlations for the dataset.

    Args:
        data (pandas.DataFrame): Input data.
        output_path (str): Path to save the heatmap image.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

# Main script
def main():
    # Example file path and parameters
    file_path = "input.csv"  # Replace with your dataset file
    heatmap_output = "correlation_heatmap.png"

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Drop non-numeric columns for PCA
    numeric_data = data.select_dtypes(include=[np.number])

    # Perform PCA
    n_components = 2  # Adjust as needed
    try:
        transformed_data = pca_numpy(numeric_data.to_numpy(), n_components)
        print("PCA completed successfully.")
        print(f"Transformed Data (first 5 rows):\n{transformed_data[:5]}")
    except Exception as e:
        print(f"Error performing PCA: {e}")

    # Generate correlation heatmap
    generate_correlation_heatmap(numeric_data, heatmap_output)

if __name__ == "__main__":
    main()
