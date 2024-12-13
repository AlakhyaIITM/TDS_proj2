import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import openai

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
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(output_file)
    plt.close()

# Function to create README.md with LLM assistance
def create_readme(file_path, analysis_summary, insights, output_prefix):
    prompt = f"""
    Write a story about the data analysis performed on {file_path}.
    Include:
    - A brief description of the data
    - The analysis steps carried out
    - Key insights discovered
    - Implications of these insights
    Use Markdown for formatting and include links to generated images ({output_prefix}_heatmap.png).
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Replace with the actual model available to you
        messages=[
            {"role": "system", "content": "You are an expert data analyst and storyteller."},
            {"role": "user", "content": prompt}
        ]
    )
    with open(f"{output_prefix}_README.md", "w") as f:
        f.write(response['choices'][0]['message']['content'])

# Function to analyze dataset and generate outputs
def analyze_dataset(file_path, output_prefix):
    try:
        df = pd.read_csv(file_path)
        numeric_data = df.select_dtypes(include=[np.number])

        principal_components, explained_variance = pca_numpy(numeric_data, num_components=2)
        pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
        pca_df.to_csv(f"{output_prefix}_pca_results.csv", index=False)

        generate_correlation_heatmap(numeric_data, f"{output_prefix}_heatmap.png")

        insights = {
            "top_principal_components": explained_variance.tolist(),
            "data_summary": numeric_data.describe().to_dict()
        }
        create_readme(file_path, "PCA and Correlation Analysis", insights, output_prefix)

        print(f"Analysis completed for {file_path}. Results saved with prefix {output_prefix}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Parallel processing for multiple datasets
def process_datasets_parallel(datasets):
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
