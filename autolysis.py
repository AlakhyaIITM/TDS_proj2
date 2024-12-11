import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# Set up the API key and proxy URL
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if openai.api_key:
    print("API Key loaded successfully!")
else:
    print("API Key not found. Please set it as an environment variable.")


def load_data(file_path):
    """Load dataset with encoding error handling."""
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


def analyze_data(data):
    """Display basic statistics and insights from the dataset."""
    print("--- Summary Statistics ---")
    print(data.describe())
    print("--- Missing Data ---")
    print(data.isnull().sum())

    # Return a summary of the data analysis
    return data.describe().to_string()


def visualize_data(data, output_folder):
    """Generate and save visualizations."""
    print("--- Generating Graphs ---")

    # Correlation heatmap
    correlation = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.title("Correlation Heatmap")
    plt.savefig(heatmap_path)
    print(f"Saved: {heatmap_path}")
    plt.close()

    # Histogram for numeric columns
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30, color="blue")
        histogram_path = os.path.join(output_folder, f"{column}_histogram.png")
        plt.title(f"Histogram of {column}")
        plt.savefig(histogram_path)
        print(f"Saved: {histogram_path}")
        plt.close()

    print("--- Graphs Saved Successfully ---")


def generate_story(data_summary):
    """Generate insights and a story using the dataset summary."""
    report_prompt = f"""
    Create a summary of the following dataset analysis:
    
    {data_summary}
    
    Include:
    - Key trends and insights
    - Any notable correlations or patterns
    - Recommendations for next steps based on the data
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Keep the model as gpt-4-mini
            messages=[
                {"role": "system", "content": "You are an assistant generating data analysis summaries."},
                {"role": "user", "content": report_prompt}
            ],
            max_tokens=500
        )

        # Extract the generated content
        story = response['choices'][0]['message']['content'].strip()
        return story

    except openai.OpenAIError as e:  # Updated to catch the new exception
        print(f"Error generating story: {e}")
        return "Story generation failed due to an error."


def save_story(story, output_folder):
    """Save the analysis story into a README.md file."""
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as file:
        file.write(story)
    print(f"Saved: {readme_path}")


def main():
    """Main script function."""
    file_path = input("Enter the path to the CSV file: ").strip()
    data = load_data(file_path)

    if data is not None:
        # Create output folder
        output_folder = create_output_folder(file_path)

        # Perform analysis and visualization
        data_summary = analyze_data(data)
        visualize_data(data, output_folder)

        # Generate story and save it to README.md
        story = generate_story(data_summary)
        save_story(story, output_folder)


if __name__ == "__main__":
    main()
