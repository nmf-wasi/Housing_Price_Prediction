# 🏠 House Price Prediction using Machine Learning

A complete end-to-end Machine Learning regression project that predicts house sale prices using the Ames Housing dataset. This project demonstrates a professional ML workflow, including data preprocessing, feature engineering, model comparison, evaluation, visualization, and model persistence.

---

## Project Overview

The objective of this project is to predict the sale price of residential houses based on various structural, geographical, and quality-related features.

Instead of training only a single model, multiple regression algorithms were implemented and compared to determine the best-performing approach.

---

## Project Structure

```
house-price-prediction/
│
├── data/
│   └── train.csv
│
├── images/
│   ├── actual_vs_predicted_*.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── best_model.txt
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── feature_importance.py
│   ├── save_model.py
│   ├── reporting.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses  **Housing Dataset**, which contains detailed information about residential properties and their corresponding sale prices.

**Target Variable**

* SalePrice

---

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* joblib

---

## Machine Learning Workflow

### 1. Data Loading

* Loaded dataset using pandas.

### 2. Data Preprocessing

* Missing value handling
* Feature engineering
* Creation of new feature:

  * `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`

### 3. Feature Transformation

* StandardScaler for numerical features
* OneHotEncoder for categorical features
* ColumnTransformer
* scikit-learn Pipeline

### 4. Train/Test Split

* 80% Training
* 20% Testing
* `random_state=42` for reproducibility

---

## Models Trained

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

---

## Evaluation Metrics

Each model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

The performance of all models was exported to:

```
results/model_comparison.csv
```

The best-performing model was automatically saved as:

```
models/best_model.pkl
```

---

## Visualizations

The project generates several visualizations:

* Actual vs Predicted plots for each model
* Feature Importance (tree-based models)
* Model Comparison chart

These plots are saved inside the `images/` directory.

---

## Results

| Model             |   R² Score |
| ----------------- | ---------: |
| Gradient Boosting | **0.9047** |
| Random Forest     |     0.8852 |
| Decision Tree     |     0.7718 |
| Linear Regression |     0.4446 |

Gradient Boosting achieved the best overall performance on the test dataset, explaining approximately **90% of the variance** in house prices.

---

## How to Run

Clone the repository:

```bash
git clone <repository-url>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/main.py
```

---

## Key Concepts Demonstrated

* Regression
* Data preprocessing
* Missing value handling
* Feature engineering
* One-hot encoding
* Feature scaling
* ColumnTransformer
* scikit-learn Pipelines
* Model comparison
* Model evaluation
* Feature importance
* Model serialization with joblib

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* XGBoost and LightGBM comparison
* Residual analysis
* SHAP values for model interpretability
* Deploy the model using Flask or FastAPI
* Build a web interface for predictions

---

## Author

**Fairuz Nasser Muhammad**

CS@SejongUniversity
