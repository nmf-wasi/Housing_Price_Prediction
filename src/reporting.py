import pandas as pd

def save_model_comparison(results, output_path):
    df = pd.DataFrame(results)
    df = df[
        ["Model", "MAE", "MSE", "RMSE", "R2"]
    ]
    df = df.sort_values(
        by="R2",
        ascending=False
    )
    df.to_csv(output_path, index=False)
    return df