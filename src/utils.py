import os 
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    
    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
       
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def save_model(model, file_path):
    """
    Save a trained model to a file using dill.
    
    Parameters:
    - model: The trained model to be saved.
    - file_path (str): The path where the model will be saved.
    
    Raises:
    - CustomException: If there is an error during saving the model.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(model, file_obj)
        print(f"Model saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
