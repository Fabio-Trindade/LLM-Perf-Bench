import logging
from src.data_structures.csv_data_format import CSVDataFormat
import pandas as pd
import os

class SingleCSVWriter:
    _df = None

    @classmethod
    def initialize(cls, data_format, file_path):
        columns = [var for var in vars(data_format)]
        cls.read_csv(file_path, columns)
        logging.info(f"CSVWriter: Initialized with columns {columns} and file path {file_path}")    

    @classmethod
    def read_csv(cls, file_path: str, columns):
        if not os.path.exists(file_path):
            logging.info(f"CSVWriter: File {file_path} does not exist. Initializing CSV.")
            cls._df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv(file_path)
            if cls._df is not None:
                raise RuntimeError("CSVWriter: DataFrame already loaded. Use write() to update it.")
            cls._df = df


    @classmethod
    def write(cls, data):
        dict = vars(data)
        if cls._df is None:
            raise RuntimeError("CSVWriter: DataFrame is not initialized. Call read_csv() first.")

        new_row = pd.DataFrame([dict])
        cls._df = pd.concat([cls._df, new_row], ignore_index=True)
    
    @classmethod
    def save(cls, file_path: str):
        logging.info(f"CSVWriter: Saving DataFrame to {file_path}")
        if cls._df is None:
            raise RuntimeError("CSVWriter: DataFrame is not initialized. Call read_csv() first.")
        
        cls._df.to_csv(file_path, index=False)
        logging.info(f"CSVWriter: DataFrame saved successfully to {file_path}")