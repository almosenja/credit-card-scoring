import os
import sys

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Initiating model training")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            seed = 42
            models = {
                "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(random_state=seed),
                "XGBoost": XGBClassifier(random_state=seed, n_jobs=-1),
                "GaussianNB": GaussianNB(),
                "DecisionTree": DecisionTreeClassifier(random_state=seed)
            }

            model_report = evaluate_models(X_train=X_train,
                                           y_train=y_train,
                                           X_test=X_test,
                                           y_test=y_test,
                                           models=models)
            
            best_model = max(model_report, key=lambda x: model_report[x]["accuracy_score"])
            selected_model = models[best_model]
            logging.info(f"Best model: {best_model}, Accuracy: {model_report[best_model]['accuracy_score']}")

            save_object(obj=selected_model, 
                        file_path=self.config.model_file_path)
            
            logging.info("Model saved successfully")

            return model_report[best_model]['accuracy_score']


        except Exception as e:
            raise CustomException(e, sys)