import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data') 
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )  

            models = {
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Decision Tree Regressor":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2', None],
                },
                'Random Forest Regressor':{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'max_features':['sqrt','log2', None],
                    'n_estimators':[8,16,32,64,128,256],
                },
                'Gradient Boosting Regressor':{
                    'loss':['squared_error','absolute_error','huber','quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.5,0.7,0.9,1.0],
                    'criterion':['friedman_mse','squared_error'],
                    'max_features':['sqrt','log2'],
                },
                'Linear Regression':{},
                'KNeighbors Regressor':{
                    'n_neighbors':[5,7,9,11],
                    'weights':['uniform','distance'],
                    'algorithm':['ball_tree','kd_tree','brute'],
                },
                'XGBoost Regressor':{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                },
                'CatBoost Regressor':{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'iterations':[30,50,100],
                },
                'AdaBoost Regressor':{
                    'n_estimators':[8,16,32,64,128,256],
                    'loss':['linear','square','exponential'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                }
            }    

            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, models=models, x_test=x_test, y_test=y_test, param=params)

            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing data: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

                    

