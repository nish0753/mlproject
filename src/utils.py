import os 
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

def evaluate_models(x_train, y_train, models, x_test, y_test,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i] 
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3) 
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_) # set best parameters to model
            model.fit(x_train, y_train) # fit model with best parameters

            model.fit(x_train, y_train) # train model

            y_train_pred = model.predict(x_train) # predict on train data
            y_test_pred = model.predict(x_test) # predict on test data

            train_model_score = r2_score(y_train, y_train_pred) # evaluate model on train data
            test_model_score = r2_score(y_test, y_test_pred) # evaluate model on test data

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
           
