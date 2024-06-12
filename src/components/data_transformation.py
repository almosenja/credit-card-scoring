import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_preprocessor

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifact", "preprocessor.pkl")

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col in self.columns:
            if X[col].dtype == "object":
                X[col] = X[col].str.extract("(\d+)").astype(float)
        return X
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Occupation column
        X["Occupation"] = X["Occupation"].str.replace("_______", "Other", regex=True)
        occupation_column = sorted(X["Occupation"].unique().tolist())
        occupation_encoded = {occupation: i for i, occupation in enumerate(occupation_column)}
        X['Occupation'] = X['Occupation'].map(occupation_encoded)

        # Credit_Mix column
        X["Credit_Mix"] = X["Credit_Mix"].replace('_', np.nan, regex=True)
        X["Credit_Mix"] = X["Credit_Mix"].ffill()
        X["Credit_Mix"] = X["Credit_Mix"].fillna(X["Credit_Mix"].mode()[0])
        credit_mix_encoded = {"Bad": 0, "Standard": 1, "Good": 2}
        X["Credit_Mix"] = X["Credit_Mix"].map(credit_mix_encoded)

        # Payment_of_Min_Amount column
        payment_min_encoded = {"No": 0, "NM": 1, "Yes": 2}
        X["Payment_of_Min_Amount"] = X["Payment_of_Min_Amount"].map(payment_min_encoded)

        # Payment_Behaviour column
        X["Payment_Behaviour"] = X["Payment_Behaviour"].replace("!@9#%8", np.nan, regex=True)
        X["Payment_Behaviour"] = X["Payment_Behaviour"].ffill()
        X["Payment_Behaviour"] = X["Payment_Behaviour"].fillna(X["Payment_Behaviour"].mode()[0])
        payment_behaviour_encoded = {
            "Low_spent_Small_value_payments": 0,
            "Low_spent_Medium_value_payments": 1,
            "Low_spent_Large_value_payments": 2,
            "High_spent_Small_value_payments": 3,
            "High_spent_Medium_value_payments": 4,
            "High_spent_Large_value_payments": 5
        }
        X["Payment_Behaviour"] = X["Payment_Behaviour"].map(payment_behaviour_encoded)

        return X

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        # self.df = pd.read_csv(df_path)

    def data_transformer(self):
        try:
            logging.info("Initiating data transformation")

            # Define columns
            columns_to_drop = ["ID", "Customer_ID", "Month", "Name", "SSN", "Monthly_Inhand_Salary", 
                               "Type_of_Loan", "Credit_History_Age"]

            categorical_columns = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

            numerical_columns = ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit", 
                                 "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Num_Bank_Accounts", 
                                 "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date", "Num_Credit_Inquiries", 
                                 "Credit_Utilization_Ratio", "Total_EMI_per_month"]


            numerical_pipeline = Pipeline(steps=[
                ('extract_numbers', NumericalTransformer(columns=numerical_columns)),
                ('impute', SimpleImputer(strategy="median"))
            ])

            # Preprocessing steps
            preprocessor = ColumnTransformer(transformers=[
                ('drop_columns', 'drop', columns_to_drop),
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                ('categorical_pipeline', CategoricalTransformer(), categorical_columns)
            ])

            logging.info("Data transformation pipeline created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")
                
            preprocessor = self.data_transformer()
            target_column = "Credit_Score"

            X_train = train_df.drop(target_column, axis=1)
            X_test = test_df.drop(target_column, axis=1)

            credit_score_encoder = {"Poor": 0, "Standard": 1, "Good": 2}
            y_train = train_df[target_column].map(credit_score_encoder)
            y_test = test_df[target_column].map(credit_score_encoder)
            logging.info("Target column encoded successfully")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Data transformed successfully")

            train_arr = np.concatenate([X_train_transformed, y_train.values.reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([X_test_transformed, y_test.values.reshape(-1, 1)], axis=1)

            save_preprocessor(preprocessor=preprocessor, 
                              file_path=self.config.preprocessor_file_path)
            logging.info("Preprocessor saved successfully")

            return train_arr, test_arr, self.config.preprocessor_file_path

        except Exception as e:
            raise CustomException(e, sys)
        