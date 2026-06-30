
def predict_model(pipeline, X_test):
    y_pred=pipeline.predict(X_test)
    return y_pred