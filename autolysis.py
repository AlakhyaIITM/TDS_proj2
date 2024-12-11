import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Load API key from environment variable
api_key = os.getenv("AIPROXY_TOKEN")

if api_key:
    print("API Key loaded successfully!")
else:
    print("API Key not found. Please set it as an environment variable.")

def load_data(file_path):
    """Load dataset with encoding error handling."""
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Use ISO-8859-1 to handle encoding issues
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

def generate_story(data):
    """Generate insights and a story using the dataset."""
    story = """
    ## Data Analysis Summary
    
    The dataset contains various columns with information on different attributes. 
    Here's a summary of the findings:
    
    - **Key Insights**: 
        - Example insights here (replace with actual analysis).
        - Trends observed across different columns.
        
    - **Recommendations**:
        - Based on the analysis, it's recommended to focus on certain patterns, such as...
        
    ### Visualizations
    - [Correlation Heatmap](./correlation_heatmap.png)
    """
    
    return story

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
        analyze_data(data)
        visualize_data(data, output_folder)

        # Generate story and save it to README.md
        story = generate_story(data)
        save_story(story, output_folder)

if __name__ == "__main__":
    main()
