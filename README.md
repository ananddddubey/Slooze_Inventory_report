

# 📊 **Credit Risk Prediction System — PySpark + ML + Power BI**

A complete **Credit Risk Prediction** project using **PySpark**, **Machine Learning**, and a fully documented **Power BI Dashboard**.
This system analyzes credit datasets from Australia, Germany, and Taiwan and predicts whether an applicant is **high risk** or **low risk**.

---

## 🗂️ **Project Structure**

```
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
│   ├── screenshots/
│       ├── overview_page.png
│       ├── demographic_insights.png
│       ├── model_performance.png
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluation.py
│
├── reports/
│   ├── Credit_Risk_Report.pdf
│
├── requirements.txt
└── README.md
```

---

# 📥 **Datasets**

This project uses 3 standard credit datasets:

### **1. Australian Credit Approval Dataset**

### **2. German Credit Dataset**

### **3. Taiwan Credit Card Default Dataset**

Each contains financial, demographic, and behavioral attributes to classify customers as **Good Credit** or **Bad Credit**.

---

# 🔧 **Technology Stack**

### **Programming & Processing**

* PySpark (MLlib, Spark SQL)
* Python
* Pandas, NumPy

### **Machine Learning Models**

* Random Forest
* Gradient Boosting
* XGBoost
* KNN
* Decision Tree
* AdaBoost
* LightGBM
* CART
* ANN (Keras / PyTorch)
* MLP (Spark MLlib)

### **Visualization**

* Power BI (Interactive Dashboard)
* Matplotlib (Model comparison graphs)

---

# ⚙️ **Project Workflow**

## **1️⃣ Data Preprocessing**

✔ Handle missing values
✔ Encode categorical variables
✔ Feature scaling
✔ Class balancing (if needed)
✔ Train/test split

---

## **2️⃣ Model Training**

All ML models are trained on all datasets:

| Model             | Australian | German | Taiwan |
| ----------------- | ---------- | ------ | ------ |
| Random Forest     | ✔          | ✔      | ✔      |
| XGBoost           | ✔          | ✔      | ✔      |
| ANN               | ✔          | ✔      | ✔      |
| KNN               | ✔          | ✔      | ✔      |
| Gradient Boosting | ✔          | ✔      | ✔      |
| Decision Tree     | ✔          | ✔      | ✔      |
| AdaBoost          | ✔          | ✔      | ✔      |
| LightGBM          | ✔          | ✔      | ✔      |
| CART              | ✔          | ✔      | ✔      |

Hyperparameter tuning performed using:

* Grid Search
* Cross-Validation (k-fold)

---

## **3️⃣ Model Evaluation Metrics**

Each model is evaluated on:

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* Log Loss
* Confusion Matrix

All results are visualized in comparison charts.




Contains:

* Dataset explanation
* ML pipeline
* Model comparison
* Power BI dashboard insights
* Conclusions

---

# 👨‍💻 **Author**

**Anand Dubey**
Research Intern | Data Analyst | ML Engineer
Python | PySpark | SQL | Power BI

