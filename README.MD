# End-to-End Machine Learning Pipeline for Student Performance Prediction

This project demonstrates a complete, production-ready machine learning pipeline that predicts student mathematics scores based on demographic and academic factors. The system combines advanced ML techniques with a user-friendly web interface to provide accurate performance predictions.

## 📊 Project Overview

**ExamInsight** is an intelligent student performance prediction system that leverages machine learning to forecast mathematics scores based on various student characteristics. The project processes student demographic data (gender, race/ethnicity, parental education), socioeconomic indicators (lunch type), academic preparation (test prep courses), and academic performance metrics (reading and writing scores) to predict mathematics achievement.

The system employs a comprehensive ML pipeline that automatically ingests data, performs sophisticated preprocessing, trains multiple regression models, and deploys the best-performing model through both programmatic APIs and an intuitive web interface. This end-to-end solution showcases modern MLOps practices including automated model selection, artifact management, and production deployment capabilities.

## 🚀 Key Features

- **🎯 Intelligent Prediction Engine:**  
  Predicts student mathematics scores (0-100) using machine learning algorithms trained on demographic, socioeconomic, and academic performance data.

- **📊 Comprehensive Data Processing:**  
  Automated pipeline that ingests student data, handles missing values, encodes categorical variables (gender, ethnicity, education levels), and scales numerical features (reading/writing scores) for optimal model performance.

- **🤖 Advanced Model Selection:**  
  Trains and evaluates multiple regression algorithms including XGBoost, CatBoost, Random Forest, Linear Regression, Lasso, Ridge, and AdaBoost. Automatically selects the best-performing model based on R² score validation.

- **🌐 Interactive Web Application:**  
  Flask-powered web interface with an intuitive form for inputting student characteristics and receiving real-time math score predictions with detailed visualization.

- **⚡ Production-Ready Pipeline:**  
  Modular architecture with automated logging, exception handling, model serialization, and artifact management for seamless deployment.

- **🔄 End-to-End Automation:**  
  Complete workflow from raw data ingestion to model deployment, with reproducible results and version-controlled artifacts.

## 📈 Dataset & Prediction Target

The model is trained on a comprehensive student performance dataset containing:

**Input Features:**
- **Demographics:** Gender, Race/Ethnicity (Groups A-E)
- **Socioeconomic:** Parental education level, Lunch type (standard/free-reduced)
- **Academic Preparation:** Test preparation course completion status
- **Academic Performance:** Reading scores, Writing scores (0-100 scale)

**Prediction Target:**
- **Mathematics Score** (0-100): The primary outcome variable representing student performance in mathematics

The dataset enables the model to understand complex relationships between demographic factors, socioeconomic status, academic preparation, and performance across different subject areas to accurately predict mathematics achievement.

## 📁 Project Structure

```
mlproject/
│
├── application.py            # Flask web application for predictions
├── templates/               # HTML templates for web interface
│   ├── index.html          # Landing page
│   └── home.html           # Prediction form and results
├── artifacts/              # Generated ML artifacts
│   ├── model.pkl          # Trained best model
│   ├── preprocessor.pkl   # Data preprocessing pipeline
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Testing dataset
│   └── raw.csv            # Original dataset
├── notebook/               # Jupyter notebooks and data
│   ├── data/stud.csv      # Raw student performance data
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/                    # Source code modules
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py # Feature preprocessing
│   │   └── model_trainer.py       # Model training and selection
│   ├── pipeline/
│   │   ├── predict_pipeline.py    # Prediction pipeline
│   │   └── train_pipeline.py      # Training pipeline
│   ├── exception.py        # Custom exception handling
│   ├── logger.py          # Logging configuration
│   └── utils.py           # Utility functions
├── requirements.txt        # Python dependencies
└── setup.py               # Package configuration
```

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nish0753/mlproject.git
   cd mlproject
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(macOS only) Install OpenMP for XGBoost:**
   ```bash
   brew install libomp
   ```

5. **Train the model (first time setup):**
   ```bash
   cd /path/to/mlproject
   PYTHONPATH=. python src/components/data_ingestion.py
   ```

6. **Run the web application:**
   ```bash
   python application.py
   ```
   
   The application will be available at `http://localhost:5001`

### 🚀 Quick Start (Web Interface)

1. Navigate to `http://localhost:5001` in your browser
2. Fill in the student information form:
   - Select demographic details (gender, ethnicity)
   - Choose parental education level
   - Select lunch type and test preparation status
   - Enter reading and writing scores (0-100)
3. Click "Predict Score" to get the mathematics score prediction
4. View results with visual gauge and detailed input summary

## 📝 How It Works

### 🔄 ML Pipeline Workflow

1. **Data Ingestion (`data_ingestion.py`):**  
   Loads raw student data from `notebook/data/stud.csv`, performs stratified train-test split (80/20), and saves datasets to `artifacts/` directory for reproducibility.

2. **Data Transformation (`data_transformation.py`):**  
   - **Numerical Features:** Applies median imputation and standard scaling to reading/writing scores
   - **Categorical Features:** Handles missing values with mode imputation, applies one-hot encoding, and scales the encoded features
   - Saves the preprocessing pipeline as `artifacts/preprocessor.pkl` for consistent inference

3. **Model Training (`model_trainer.py`):**  
   - Trains multiple regression models: Linear Regression, Lasso, Ridge, Random Forest, Decision Tree, XGBoost, CatBoost, AdaBoost, K-Neighbors
   - Performs hyperparameter tuning using GridSearchCV with 3-fold cross-validation
   - Evaluates models using R² score on test set
   - Saves the best-performing model as `artifacts/model.pkl`

### 🌐 Web Application

The Flask web application (`application.py`) provides an intuitive interface for making predictions:

- **Homepage Route (`/`):** Renders the landing page
- **Prediction Route (`/predictdata`):** 
  - **GET:** Displays the prediction form with dropdowns for categorical features and number inputs for scores
  - **POST:** Processes form data, runs prediction pipeline, and displays results with input summary and visual score gauge

### 🎯 Prediction Process

1. User inputs student characteristics through the web form
2. `CustomData` class structures the input into a pandas DataFrame
3. `PredictPipeline` loads the saved preprocessor and model
4. Data is transformed using the preprocessing pipeline
5. Model generates math score prediction (0-100)
6. Results are displayed with a circular gauge visualization and input summary

## 📊 Results & Performance

### 🎯 Model Performance
- **Best Model:** The pipeline automatically selects the highest-performing regression model based on R² score validation
- **Evaluation Metric:** R² (coefficient of determination) on test set
- **Prediction Range:** Mathematics scores from 0 to 100
- **Model Artifacts:** All trained models and preprocessing pipelines are saved in `artifacts/` directory

### 🔍 Key Insights
- **Feature Importance:** Reading and writing scores are strong predictors of mathematics performance
- **Demographic Impact:** Parental education level and socioeconomic factors significantly influence academic outcomes
- **Academic Preparation:** Test preparation courses show measurable impact on score predictions

### 📁 Generated Artifacts
```
artifacts/
├── model.pkl          # Best performing trained model
├── preprocessor.pkl   # Data preprocessing pipeline
├── train.csv         # Training dataset (80%)
├── test.csv          # Testing dataset (20%)
└── raw.csv           # Original complete dataset
```

### 🌐 Web Application Features
- **Real-time Predictions:** Instant mathematics score predictions through web interface
- **Visual Results:** Circular gauge visualization displaying predicted scores
- **Input Validation:** Form validation ensuring data quality and range constraints
- **Responsive Design:** Mobile-friendly interface with modern styling

## 🙏 Acknowledgements

Special thanks to [Krish Naik](https://www.linkedin.com/in/krishnaik06/) for his invaluable guidance and educational content.

---

**GitHub:** [nish0753/mlproject](https://github.com/nish0753/mlproject)

---

## ⚠️ Troubleshooting AWS Deployments

If you delete a file (like `app.py`) from your repo and AWS Elastic Beanstalk or CodePipeline still references it, follow these steps:

1. **Verify on GitHub** that the file is truly deleted from your main branch.
2. **Delete old application versions** in the Elastic Beanstalk console:
   - Go to your application.
   - Click on **Application versions** in the sidebar.
   - Delete all old versions except the most recent one.
3. **Trigger a fresh deployment** by making a small change (e.g., edit `README.md`), commit, and push.
4. **If the problem persists**, delete and recreate your Elastic Beanstalk environment and CodePipeline for a clean start.

This ensures AWS uses only the current files in your repo and stops referencing deleted files.