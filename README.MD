# End-to-End Machine Learning Pipeline

This project demonstrates a complete, production-ready machine learning pipeline, from data ingestion to model training and evaluation, using Python and popular ML libraries. The workflow is fully automated, modular, and version-controlled.

## 🚀 Features

- **Data Ingestion:**  
  Reads raw data, performs train-test split, and saves the results for reproducibility.

- **Data Transformation:**  
  Handles missing values, encodes categorical variables, and scales numerical features using robust pipelines. The preprocessor is saved as a `.pkl` file for easy reuse.

- **Model Training & Selection:**  
  Trains and compares multiple regression models (XGBoost, CatBoost, Random Forest, etc.), automatically selects the best model based on R² score, and saves it for deployment.

- **Automation & Logging:**  
  All steps are automated with clear logging and exception handling.

- **Version Control:**  
  The entire project is tracked with Git and available on GitHub.

## 📁 Project Structure

```
ML Project/
│
├── artifacts/                # Saved models, preprocessors, and data splits
├── notebook/                 # Jupyter notebooks and raw data
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── requirements.txt
└── README.md
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
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(macOS only) Install OpenMP for XGBoost:**
   ```bash
   brew install libomp
   ```

5. **Run the pipeline:**
   ```bash
   PYTHONPATH=. python src/components/data_ingestion.py
   ```

## 📝 How It Works

- **Data Ingestion:**  
  Loads raw data from `notebook/data/stud.csv`, splits into train/test, and saves to `artifacts/`.

- **Data Transformation:**  
  Applies preprocessing (imputation, encoding, scaling) and saves the preprocessor as `artifacts/preprocessor.pkl`.

- **Model Training:**  
  Trains several regression models, evaluates them, and saves the best model as `artifacts/model.pkl`. The R² score is printed at the end.

## 📊 Results

- The pipeline prints the R² score of the best model on the test set.
- All artifacts (preprocessor, model, data splits) are saved in the `artifacts/` directory.

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