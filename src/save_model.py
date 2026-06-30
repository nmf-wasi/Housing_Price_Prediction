import joblib

def save_model(model, model_path):
    joblib.dump(model,model_path)