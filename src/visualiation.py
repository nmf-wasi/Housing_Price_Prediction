import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_test, y_pred, model_name):

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(f"{model_name}: Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(
        f"../images/{model_name.lower().replace(' ','_')}_prediction.png"
    )

    plt.close()

def plot_model_comparison(results_df):

    plt.figure(figsize=(8,5))

    plt.bar(
        results_df["Model"],
        results_df["RMSE"]
    )

    plt.ylabel("RMSE")

    plt.title("Model Comparison")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig("../images/model_comparison.png")

    plt.close()