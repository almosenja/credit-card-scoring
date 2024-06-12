import os
import sys
import joblib
from src.exception import CustomException

def save_preprocessor(preprocessor, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(preprocessor, file_path)
    except Exception as e:
        raise CustomException(e, sys)