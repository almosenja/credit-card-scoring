import os
import sys
import joblib
from src.exception import CustomException
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = {
                "accuracy_score": accuracy_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred)
            }
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)