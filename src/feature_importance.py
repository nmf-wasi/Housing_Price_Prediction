import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(pipeline):
    model = pipeline.named_steps['model']
    if not hasattr(model, 'feature_importance'):
        print("Model doesn't have 'feature_importance' attribute")
        return
    feature_names=pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model.feature_importances_
    importance_df=pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )
    importance_df=importance_df.sort_values(by='Importance', ascending=False)
    top15=importance_df.head(15)
    plt.figure(figsize=(15,6))
    plt.bar(top15['Feature'], top15['Importance'], color='red')
    plt.gca().invert_yaxis()
    plt.title("Top 15 Most Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(
        "../images/feature_importance.png",
        dpi=300
    )
    plt.show()