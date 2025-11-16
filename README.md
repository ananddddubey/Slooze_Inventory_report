📌 Overview

This project is a Credit Risk Prediction System built using PySpark + MLlib, designed to evaluate an applicant’s creditworthiness using multiple machine learning models. The system processes three datasets — Australian, German, and Taiwan credit datasets — and compares the performance of various classification algorithms.

Along with machine learning modeling, an interactive Power BI dashboard is created to visually analyze risk factors, dataset distribution, and model insights.

🗂️ Project Structure
Credit-Risk-Prediction/
│
├── data/
│   ├── australian.csv
│   ├── german.csv
│   ├── taiwan.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_comparison.ipynb
│
├── powerbi/
│   ├── credit_risk_dashboard.pbix
│   ├── dashboard_screenshots/
│
├── reports/
│   ├── Credit_Risk_Report.pdf
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluation.py
│
└── README.md

📥 Input Datasets

The project uses three widely used financial datasets:

Australian Credit Approval Dataset

German Credit Dataset

Taiwan Credit Card Default Dataset

Each dataset includes demographic, financial, and credit history attributes.

🔧 Technologies Used
Data Engineering & ML

PySpark (Spark SQL, Spark MLlib)

Python

Pandas, NumPy, Matplotlib

Machine Learning Models

Multilayer Perceptron (MLP)

Random Forest

Gradient Boosting

XGBoost

KNN

ANN (PyTorch/Keras)

Decision Tree

AdaBoost

LightGBM

CART

Visualization

Power BI

Matplotlib (for model comparison charts)

⚙️ Steps Performed
1️⃣ Data Preprocessing

Handling missing values

Feature engineering

One-hot encoding

Label indexing

Train-test split

Scaling (MinMaxScaler / StandardScaler)

2️⃣ Model Training

Each dataset is trained using all algorithms.
Hyperparameter tuning is performed using:

Grid Search

Cross-validation

3️⃣ Model Evaluation

Metrics captured:

Accuracy

Precision

Recall

F1-score

AUC

Log Loss

Confusion Matrix

A visual comparison graph is generated for all algorithms.

📊 Power BI Dashboard (Included)

A fully designed Power BI dashboard is included in the project folder (powerbi/credit_risk_dashboard.pbix).

Dashboard Sections
1️⃣ Overview Page

Total applicants

Default vs Non-default ratio

Risk distribution

Credit score segmentation

2️⃣ Customer Demographics

Age distribution

Gender split

Education level

Marital status

3️⃣ Financial Insights

Loan amount distribution

Income analysis

Purpose of credit

Historical repayment patterns

4️⃣ Model Insights

Accuracy of each model

AUC comparison chart

Best-performing model indicator

5️⃣ Filters

By dataset (Australian / German / Taiwan)

By income category

By age group

By risk level

📈 Model Performance Visualization

A comparative bar graph is generated showing:

Accuracy

F1 Score

AUC

This helps in identifying the best model for each dataset.

🧾 Report

A complete PDF report is included under /reports/Credit_Risk_Report.pdf, covering:

Data description

Preprocessing methods

Model tuning

Evaluation

Power BI dashboard explanation

Final conclusions
