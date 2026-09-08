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

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Split training and test input data"
            )

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Models
            models = {

                "Random Forest":
                    RandomForestRegressor(
                        random_state=42
                    ),

                "Decision Tree":
                    DecisionTreeRegressor(
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    ),

                "Linear Regression":
                    LinearRegression(),

                "K-Neighbors Regressor":
                    KNeighborsRegressor(),

                "XGBRegressor":
                    XGBRegressor(
                        random_state=42
                    ),

                "CatBoosting Regressor":
                    CatBoostRegressor(
                        verbose=False,
                        random_seed=42
                    ),

                "AdaBoost Regressor":
                    AdaBoostRegressor(
                        random_state=42
                    ),
            }

            # Hyperparameters
            params = {

                "Decision Tree": {

                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson"
                    ],

                    "max_depth": [
                        None,
                        3,
                        5,
                        7,
                        10
                    ]
                },

                "Random Forest": {

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ],

                    "max_depth": [
                        None,
                        5,
                        10,
                        15
                    ]
                },

                "Gradient Boosting": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {

                    "n_neighbors": [
                        3,
                        5,
                        7,
                        9,
                        11
                    ],

                    "weights": [
                        "uniform",
                        "distance"
                    ]
                },

                "XGBRegressor": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "CatBoosting Regressor": {

                    "depth": [
                        6,
                        8,
                        10
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "iterations": [
                        30,
                        50,
                        100
                    ]
                },

                "AdaBoost Regressor": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.5,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                }
            }

            # Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Get best model score
            best_model_score = max(
                model_report.values()
            )

            # Get best model name
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            # Get best model
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}"
            )

            logging.info(
                f"Best model R2 score: {best_model_score}"
            )

            # Check model performance
            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found",
                    sys
                )

            logging.info(
                "Best found model on both training and testing dataset"
            )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final prediction
            predicted = best_model.predict(X_test)

            # Final R2 score
            r2_square = r2_score(
                y_test,
                predicted
            )

            return r2_square

        except Exception as e:

            raise CustomException(e, sys)