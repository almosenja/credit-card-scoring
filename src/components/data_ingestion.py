import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('always')


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifact", "train.csv")
    test_data_path: str=os.path.join("artifact", "test.csv")
    raw_data_path: str=os.path.join("artifact", "data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion")

            df = pd.read_csv("data/train.csv", low_memory=False)
            logging.info("Data loaded successfully")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Data saved successfully")

            train, test = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split successfully")

            train.to_csv(self.config.train_data_path, index=False, header=True)
            test.to_csv(self.config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")

            return self.config.train_data_path, self.config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformer = DataTransformation()
    train_arr, test_arr = data_transformer.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))