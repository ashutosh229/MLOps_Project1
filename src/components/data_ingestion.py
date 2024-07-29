from dataclasses import dataclass
from src.exception.exception import customexception
import sys
from src.utils.utils import save_object
import os
from src.logger.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split




@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw_data.csv")
    train_data_path:str = os.path.join("artifacts","train_data.csv")
    test_data_path:str = os.path.join("artifacts","test_data.csv")
    

class DataIngestion:
    def __init__(self):
        self.configObject = DataIngestionConfig()
    
    def initializeDataIngestion(self):
        logging.info("Data Ingestion Started")
        try:
            data = pd.read_csv("C:/Users/akuma/Downloads/gemstonedata/train.csv")
            os.makedirs(os.path.dirname(os.path.join(self.configObject.raw_data_path)))
            data.to_csv(self.configObject.raw_data_path,index=False)
            
            train_data,test_data=train_test_split(data, test_size=0.25)
            train_data.to_csv(self.configObject.train_data_path,index=False)
            test_data.to_csv(self.configObject.test_data_path,index=False)
            
            logging.info("Data ingestion is completed")
            return (
                self.configObject.train_data_path,
                self.configObject.test_data_path
            )
        except Exception as e:
            logging.info("Error occured during data ingestion")
            raise customexception(e,sys)
        

        