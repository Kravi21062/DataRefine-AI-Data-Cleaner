DataRefine — AI-Powered Data Cleaning and Profiling
DataRefine is a Streamlit-based tool for exploring, cleaning, and profiling messy datasets. It helps users inspect data quality, apply automated or manual cleaning operations, review before-and-after results, generate insights, and export cleaned data with PDF reports.
Live application: https://datarefine-ai-data-cleaner.streamlit.app/
Key Features
Upload CSV, XLS, and XLSX datasets
Interactive data overview and data-quality dashboard
AI-assisted cleaning suggestions, with a rule-based fallback
Missing-value handling
Duplicate detection and removal
Outlier detection using IQR and Z-score methods
Encoding and feature-scaling utilities
Text-standardization utilities
Before-and-after comparison analytics
AI-generated insights and visualizations
Export cleaned datasets
Generate PDF reports
Undo support through cleaning snapshot history
How It Works
Upload a dataset.
Review the dataset overview and quality metrics.
Request cleaning suggestions.
Apply smart cleaning or choose manual cleaning operations.
Review the before-and-after comparison.
Explore generated insights and visualizations.
Export the cleaned dataset and, if needed, generate a PDF report.
System Architecture
flowchart LR
    U["User"] --> UI["Streamlit UI - app.py"]

    UI --> DU["data_utils.py"]
    UI --> CU["chart_utils.py"]
    UI --> PU["pdf_utils.py"]

    DU --> SK["scikit-learn"]
    DU --> PD["Pandas + NumPy"]

    CU --> PL["Plotly"]

    PU --> RL["ReportLab + Matplotlib"]

    DU --> OL["Ollama API (Optional)"]
Cleaning Workflow
flowchart TD
    A["Start application"] --> B["Upload dataset"]
    B --> C["Store original and working data"]
    C --> D["Generate data-quality overview"]
    D --> E{"Choose cleaning mode"}
    E -->|Smart cleaning| F["Generate suggestions and run supported automatic cleaning"]
    E -->|Manual cleaning| G["Apply selected cleaning operations"]
    F --> H["Update working dataset and save history"]
    G --> H
    H --> I["Compare before and after"]
    H --> J["Generate insights"]
    I --> K["Export CSV/XLSX"]
    J --> L["Generate PDF report"]
AI Suggestions and Fallback
DataRefine can use a locally running Ollama model to generate cleaning suggestions. The configured default model is phi3.
flowchart TD
    A["Build dataset profile"] --> B{"Is Ollama reachable?"}
    B -->|Yes| C["Call configured Ollama model"]
    C --> D{"Did it return valid JSON?"}
    D -->|Yes| E["Use AI suggestions"]
    D -->|No| F["Use rule-based suggestions"]
    B -->|No| F
    E --> G["Return suggestions"]
    F --> G
If Ollama is unavailable or its response cannot be used, DataRefine falls back to rule-based suggestions.
Outlier Detection
The current version documents these outlier detection methods:
IQR: uses quartiles to calculate per-column bounds.
Z-score: flags values whose absolute Z-score exceeds the configured threshold.
Note: Isolation Forest is not implemented in the current version.
Project Structure
DataRefine-AI-Data-Cleaner/
├── app.py
├── data_utils.py
├── chart_utils.py
├── pdf_utils.py
├── theme.py
├── requirements.txt
├── PROJECT_DOCS.md
└── README.md
Core Modules
File
Responsibility
app.py
Streamlit interface, session state, cleaning workflow, and export coordination
data_utils.py
Dataset loading, profiling, cleaning operations, AI suggestions, and insight generation
chart_utils.py
Plotly themes, dashboard charts, comparison analytics, and quality gauges
pdf_utils.py
PDF report generation, summaries, tables, and chart rendering
theme.py
Application theme configuration
Installation
Requirements
Python installed on your system
Git
The project dependencies listed in requirements.txt
1. Clone the repository
git clone https://github.com/Kravi21062/DataRefine-AI-Data-Cleaner.git
cd DataRefine-AI-Data-Cleaner
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
If PowerShell does not allow script activation, use Command Prompt:
venv\Scripts\activate.bat
Linux / macOS:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Run the Application
streamlit run app.py
Then open the local address shown in your terminal. The default Streamlit address is:
http://localhost:8501
Optional: Configure Ollama
Ollama is optional. It enables local model-based cleaning suggestions when it is available.
1. Install and start Ollama
Install Ollama using its official installation instructions and make sure the local service is running.
2. Set the environment variables
Windows PowerShell:
$env:OLLAMA_URL="http://localhost:11434/api/generate"
$env:OLLAMA_MODEL="phi3"
Linux / macOS:
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="phi3"
If Ollama is unavailable, the application uses its rule-based suggestion fallback.
Main Data Objects
Object
Purpose
raw_df
Original uploaded dataset
clean_df
Working dataset after cleaning operations
cleaning_history
Snapshots used for undo/restore functionality
Current Behavior and Limitations
The documented smart-cleaning pipeline handles missing values, removes duplicate rows, and applies IQR-based outlier filtering.
The outlier detection utilities support IQR and Z-score methods.
Isolation Forest is not implemented in the current version.
If icon assets are missing, fallback rendering is used.
PDF export handles datasets with empty columns.
Technology Stack
Area
Technologies
User interface
Streamlit
Data processing
Python, Pandas, NumPy
Machine learning utilities
scikit-learn
Visualization
Plotly, Matplotlib
PDF reporting
ReportLab
Spreadsheet support
OpenPyXL, xlrd
Optional local AI
Ollama (phi3)
Documentation
For more detailed implementation and method-level information, see PROJECT_DOCS.md.
Planned Improvements
Potential future enhancements include:
Isolation Forest outlier detection
Multi-file dataset joins
Scheduled cleaning workflows
Database connectors
Collaborative data-cleaning features
Additional cloud deployment options
LLM-based anomaly explanations
These are planned improvements and are not described as available features in the current release.
License
The original project documentation describes this project as intended for educational and research purposes. No specific open-source license is identified in this README.
Before presenting the repository as open source for reuse, add a LICENSE file that clearly states the permissions and conditions you want to apply. Until then, users should not assume that they have permission to copy, modify, or redistribute the code.
