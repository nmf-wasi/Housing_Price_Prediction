import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split

from data_processing import (
    load_data,
    preprocess,
    build_preprocessor
)
from reporting import save_model_comparison
from train import (
    build_pipeline,
    train_model
)
from predict import predict_model
from evaluation import evaluate_model
from feature_importance import plot_feature_importance
from save_model import save_model
from visualiation import plot_actual_vs_predicted, plot_model_comparison

# Load and preprocess data
X, y = load_data("../data/train.csv")
X = preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
preprocessor = build_preprocessor(X_train)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

best_pipeline = None
best_model_name = None
best_r2 = float("-inf")

# Train every model

for name, model in models.items():

    print(f"\n{name}\n")
    pipeline = build_pipeline(
        preprocessor,
        model
    )
    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )
    y_pred = predict_model(pipeline,X_test)

    metrics = evaluate_model(
        y_test,
        y_pred)
    metrics["Model"] = name
    plot_actual_vs_predicted(
        y_test,
        y_pred,
        name
    )
    results.append(metrics)

    # Save best model
    if metrics["R2"] > best_r2:
        best_r2 = metrics["R2"]
        best_pipeline = pipeline
        best_model_name = name
    # Feature Importance

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(pipeline)

# Model Comparison
results_df = pd.DataFrame(results)
results_df = save_model_comparison(
    results,
    "../results/model_comparison.csv"
)

plot_model_comparison(results_df)
plot_model_comparison(results_df)

# Save Best Model
save_model(
    best_pipeline,
    "../models/best_model.pkl"
)
with open("../results/best_model.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"R2 Score: {best_r2:.4f}\n")

print(f"\nBest Model: {best_model_name}")
print(f"Best R²: {best_r2:.4f}")