# Autolysis - Automated Data Analysis and Visualization Tool

Autolysis is a Python-based tool that streamlines the process of exploring, analyzing, and visualizing datasets. This application allows you to upload CSV files and generates meaningful insights and visualizations.

---

## Features

- Automatically handles encoding issues while loading CSV files.
- Analyzes and provides statistical summaries of the dataset.
- Generates visualizations, including:
  - Correlation heatmaps.
  - Histograms for numeric columns.
  - Insights into missing data.
- Stores all visual outputs in a dedicated folder named after the CSV file.
- API integration (via `AIPROXY_TOKEN`) for potential enhancements.

---

## Requirements

- Python 3.7+
- Libraries: pandas, seaborn, matplotlib, fastapi, uvicorn

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## API Key Setup

Set your API key as an environment variable named `AIPROXY_TOKEN`.  
Example:
- **Windows**:
  ```cmd
  set AIPROXY_TOKEN=your_actual_api_key
  ```
- **Linux/Mac**:
  ```bash
  export AIPROXY_TOKEN=your_actual_api_key
  ```

---

## Usage

1. Clone the repository.
2. Set the API key environment variable as described above.
3. Run the application:
   ```bash
   python autolysis.py
   ```
4. Navigate to the FastAPI interface to upload your CSV file and view outputs.

---

## Example Workflow

1. Upload a CSV file through the API interface.
2. The application:
   - Loads the data.
   - Creates an output folder named after the file.
   - Generates visualizations and saves them.
3. Retrieve visualizations and statistical insights from the output folder.

---

## Outputs

- Correlation heatmaps
- Numeric column histograms
- Missing data summaries

All visualizations are saved as `.png` files in the output folder.

---

## Contributing

Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request.

---

## License

This project is licensed under the MIT License.