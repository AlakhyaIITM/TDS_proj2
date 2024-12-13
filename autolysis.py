import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load dataset
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

# Function to save outputs to the appropriate folder
def save_output(output_path, data, is_visual=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if is_visual:
        plt.savefig(output_path)
        plt.close()
    else:
        with open(output_path, 'w') as f:
            f.write(data)

# Data summary function
def generate_data_summary(df):
    summary = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
    }
    return summary

# Advanced analysis with correlation and PCA
def perform_advanced_analysis(df):
    insights = {}
    # Correlation analysis
    correlations = df.corr()
    insights["correlations"] = correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm")
    save_output("goodreads/correlation_heatmap.png", None, is_visual=True)

    # PCA analysis
    numeric_data = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data.dropna())
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PCA1', y='PCA2', data=df)
        save_output("goodreads/pca_scatterplot.png", None, is_visual=True)
    return insights

# Narrative generation using OpenAI
def generate_narrative(data_summary, insights):
    summary_str = f"Data Summary: {data_summary}\nKey Insights: {insights}"
    prompt = f"""
    Generate a detailed Markdown report:
    {summary_str}
    - Highlight trends and anomalies.
    - Provide actionable insights.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000
    )
    return response['choices'][0]['text'].strip()

# Main workflow
def main(file_path):
    df = load_dataset(file_path)
    data_summary = generate_data_summary(df)
    insights = perform_advanced_analysis(df)
    narrative = generate_narrative(data_summary, insights)

    # Save outputs
    save_output("goodreads/data_summary.txt", str(data_summary))
    save_output("goodreads/narrative.md", narrative)

# Run the script
if __name__ == "__main__":
    dataset_path = "dataset.csv"  # Replace with your dataset path
    try:
        main(dataset_path)
    except Exception as e:
        print(f"Error: {e}")
