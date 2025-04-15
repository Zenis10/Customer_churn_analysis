# Customer Churn Prediction with Random Forest

This project focuses on analyzing and predicting customer churn using the **Telco Customer Churn dataset**. The goal is to build a machine learning model that can accurately identify customers who are likely to churn based on their service usage patterns, demographics, and account information.

---

## 📁 Dataset

The dataset used is **Telco Customer Churn** and contains 7043 customer records with 21 features such as:

- Customer demographics (gender, SeniorCitizen, Partner, Dependents)
- Services signed up (PhoneService, InternetService, etc.)
- Account information (tenure, MonthlyCharges, TotalCharges)
- Target: `Churn` (Yes/No)

---

## 📌 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **RandomForestClassifier**

---

## 🧾 Steps Followed

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Load and Explore the Data

- Read the CSV file using `pandas`
- Checked for missing values and summary statistics
- Visualized churn distribution using `seaborn`

### 3. Data Preprocessing

- Encoded categorical variables using `LabelEncoder`
- Split data into training and testing sets (80/20 split)
- Applied `StandardScaler` to normalize numerical features

### 4. Model Building

- Used **Random Forest Classifier** from `sklearn.ensemble`
- Trained the model on the training data
- Made predictions on the test set

### 5. Evaluation Metrics

- Confusion Matrix:

  ```
  [[946  90]
   [192 181]]
  ```

- Classification Report:

  ```
                precision    recall  f1-score   support

           0       0.83      0.91      0.87      1036
           1       0.67      0.49      0.56       373

    accuracy                           0.80      1409
   macro avg       0.75      0.70      0.72      1409
  weighted avg       0.79      0.80      0.79      1409
  ```

- Accuracy: **80%**

- Plotted a heatmap of the confusion matrix for better visualization.

---

## 📊 Visualizations

- Churn Distribution (`sns.countplot`)
- Confusion Matrix (`sns.heatmap`)

---

## 📌 Conclusion

- The Random Forest model achieved an accuracy of **80%**, with decent performance in detecting churned customers.
- Feature scaling and label encoding significantly contributed to the model's effectiveness.
- This model can be further improved using techniques like SMOTE, feature engineering, hyperparameter tuning, and trying different algorithms (e.g., XGBoost, SVM).

---

## 🔮 Future Improvements

- Handle class imbalance with SMOTE
- Hyperparameter tuning with GridSearchCV
- Add more domain-specific feature engineering
- Model comparison using multiple classifiers

---

## 🧠 Author

- **Your Name**
- [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

## 📁 Folder Structure

```
├── Telco-Customer-Churn.csv
├── churn_analysis.ipynb
└── README.md
```

---
