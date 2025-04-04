# End-to-End MLOps for Time Series Forecasting

This repository provides a comprehensive framework for managing the entire Machine Learning lifecycle, specifically tailored for Time Series data. By adhering to MLOps best practices, this project streamlines workflows from data ingestion and preprocessing to hyperparameter optimization, model training, and deployment.

While the primary focus is on Time Series data, the scripts and methodologies in this repository are generalizable and can be adapted for any Machine Learning project involving tabular data.

## Key Features

- **Complete ML Lifecycle**: Covers all phases from data preparation to model deployment.
- **Modular Codebase**: Easily transferable to other ML projects.
- **MLOps Best Practices**: Ensures reproducibility, scalability, and maintainability.

## Tech Stack

The project leverages the following technologies:

- **Python 3.10.11+**: The core programming language used throughout the project.
- **Pandas**: For data manipulation and preprocessing.
- **AWS**: Cloud provider for scalable infrastructure and deployment.
- **Scikit-Learn API**: Provides a consistent interface for Machine Learning estimators.
- **Optuna**: For advanced Hyperparameter Optimization (HPO).
- **MLflow**: For tracking experiments, models, and for model deployment.

## Project Highlights

1. **Data Ingestion**

   - Scripts to automate the collection and validation of raw data.

2. **Data Preprocessing**

   - Tools to clean, transform, and prepare data for Time Series analysis.

3. **Hyperparameter Optimization**

   - Integration with Optuna for efficient and scalable parameter tuning.

4. **Model Training**

   - Leverages Scikit-Learn's API to train models.

5. **Model Tracking and Experimentation**

   - Utilizes MLflow to log metrics, parameters, and artifacts for reproducibility.

6. **Model Deployment**

   - Flask-based deployment for serving models as RESTful APIs.

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**

   Ensure you have Python 3.10 installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure AWS**

   Set up your AWS credentials to use cloud services for storage or deployment.

7. **Run the Pipeline**

   TODO
   <!--
   Execute the script `main.py` script to run the pipeline
   
   ```
   python main.py
   -->
   ```
