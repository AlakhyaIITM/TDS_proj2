import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import argparse
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use the Agg backend for generating images without displaying them


# Set up the API key and proxy URL
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if openai.api_key:
    print("API Key loaded successfully!")
else:
    print("API Key not found. Please set it as an environment variable.")


def load_data(file_path):
    """Load dataset with encoding error handling."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at path {file_path}")
        return None

    if not file_path.endswith(".csv"):
        print("Error: The file provided is not a CSV.")
        return None

    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Handle encoding issues
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_output_folder(file_path):
    """Create a folder named after the CSV file (excluding .csv) for saving images."""
    folder_name = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def handle_missing_data(data):
    """Handle missing data by imputation or removal."""
    # Handle non-numeric columns by attempting to convert them to numeric, setting errors='coerce' to turn invalid entries into NaN
    for column in data.select_dtypes(include=[object]).columns:
        try:
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Convert non-numeric columns to NaN
        except Exception as e:
            print(f"Error converting column {column}: {e}")

    # Fill missing data (NaN) with the mean of the column
    return data.apply(pd.to_numeric, errors='coerce').fillna(data.mean(), axis=0)


def analyze_data(data):
    """Display advanced statistics and insights from the dataset."""
    try:
        print("--- Summary Statistics ---")
        summary = data.describe()
        print(summary)

        # Identify and display missing data
        print("--- Missing Data ---")
        print(data.isnull().sum())

        # Remove outliers using z-scores (optional: only for numeric columns)
        z_scores = np.abs((data.select_dtypes(include=['float64', 'int64']) - data.mean()) / data.std())
        data_no_outliers = data[(z_scores < 3).all(axis=1)]  # Keep rows where all z-scores are < 3
        print(f"Data after outlier removal: {data_no_outliers.shape}")

        # Additional insights: checking correlations
        correlation = data_no_outliers.corr()
        print("--- Correlation Matrix ---")
        print(correlation)

        return summary.to_string(), data_no_outliers, correlation
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return "", data, None


def visualize_data(data, output_folder):
    """Generate and save enhanced visualizations."""
    print("--- Generating Graphs ---")

    # Correlation heatmap
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if not numeric_data.empty:
        correlation = numeric_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation coefficient'})
        heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_path)
        print(f"Saved: {heatmap_path}")
        plt.close()
    else:
        print("No numeric columns found for correlation heatmap.")

    # Pairplot (scatter plot matrix) to visualize relationships
    if len(numeric_data.columns) > 1:  # Ensure there are at least two numeric columns
        plt.figure(figsize=(8, 6))
        sns.pairplot(numeric_data)
        pairplot_path = os.path.join(output_folder, "pairplot.png")
        plt.title("Pairplot of Numeric Columns")
        plt.savefig(pairplot_path)
        print(f"Saved: {pairplot_path}")
        plt.close()

    # Scatter plots for key correlations
    if len(numeric_data.columns) > 1:
        for i, column1 in enumerate(numeric_data.columns):
            for column2 in numeric_data.columns[i+1:]:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=data, x=column1, y=column2)
                plt.xlabel(column1)
                plt.ylabel(column2)
                scatter_path = os.path.join(output_folder, f"scatter_{column1}_vs_{column2}.png")
                plt.title(f"Scatter plot of {column1} vs {column2}")
                plt.savefig(scatter_path)
                print(f"Saved: {scatter_path}")
                plt.close()

    print("--- Graphs Saved Successfully ---")


def generate_story(data_summary, correlation_data, data):
    """Generate detailed insights and recommendations based on dataset summary."""
    if len(data_summary) > 1000:  # Avoid exceeding token limits
        data_summary = data_summary[:1000] + "..."

    # Create a dynamic prompt based on the correlation data and key insights
    correlation_insights = correlation_data.to_string() if correlation_data is not None else "No significant correlations found."

    report_prompt = f"""
    Create a detailed analysis of the following dataset:

    {data_summary}

    Insights:
    - Key trends and patterns observed in the dataset
    - Correlations identified in the correlation matrix:
    {correlation_insights}
    - Recommendations for further analysis or actions based on the findings:
      (e.g., actions to improve data quality, recommendations for stakeholders)
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Ensure using the right model
            messages=[{"role": "system", "content": "You are an assistant generating detailed data analysis summaries."},
                      {"role": "user", "content": report_prompt}],
            max_tokens=600  # Increase token limit for a more detailed story
        )

        # Extract and return the story
        story = response['choices'][0]['message']['content'].strip()
        return story

    except openai.OpenAIError as e:
        print(f"Error generating story: {e}")
        return "Story generation failed due to an error."


def save_story(story, output_folder):
    """Save the generated story into a README.md file."""
    readme_path = os.path.join(output_folder, "README.md")
    try:
        with open(readme_path, "w") as file:
            file.write(story)
        print(f"Saved: {readme_path}")
    except Exception as e:
        print(f"Error saving story: {e}")


def main():
    """Main script function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process CSV file for analysis.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    
    args = parser.parse_args()

    file_path = args.file_path  # Use the file path from the argument
    data = load_data(file_path)

    if data is not None:
        # Handle missing data
        data = handle_missing_data(data)

        # Create output folder
        output_folder = create_output_folder(file_path)

        # Perform analysis and visualization
        data_summary, data_no_outliers, correlation_data = analyze_data(data)
        visualize_data(data_no_outliers, output_folder)

        # Generate and save the analysis story
        story = generate_story(data_summary, correlation_data, data_no_outliers)
        save_story(story, output_folder)


if __name__ == "__main__":
    main()
