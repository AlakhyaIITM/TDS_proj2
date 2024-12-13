import os
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def process_dataset(dataset_path):
    try:
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Perform PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data.select_dtypes(include=[float, int]))
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        # Create a correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        heatmap_path = os.path.splitext(dataset_path)[0] + "_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()

        # Generate README content
        readme_content = f"""
# Dataset Analysis: {os.path.basename(dataset_path)}

## Principal Component Analysis (PCA)
- PCA Components:
  - PC1: {pca.explained_variance_ratio_[0]:.2f}
  - PC2: {pca.explained_variance_ratio_[1]:.2f}

## Correlation Heatmap
- Heatmap saved at: {heatmap_path}
"""

        # Write README file
        output_dir = os.path.dirname(dataset_path)
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as file:
            file.write(readme_content)
        print(f"README.md created successfully for {dataset_path} at {readme_path}")

    except Exception as e:
        print(f"Error processing dataset {dataset_path}: {e}")

if __name__ == "__main__":
    # List of dataset paths
    datasets = [
        "/path/to/dataset1.csv",
        "/path/to/dataset2.csv",
        "/path/to/dataset3.csv",
    ]

    # Ensure output directories exist
    for dataset_path in datasets:
        output_dir = os.path.dirname(dataset_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Process datasets concurrently
    with ThreadPoolExecutor() as executor:
        executor.map(process_dataset, datasets)

    print("Processing complete.")
