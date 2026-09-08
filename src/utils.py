import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params
):
    try:
        report = {}

        for model_name in models:

            model = models[model_name]
            param_grid = params.get(model_name, {})

            # If hyperparameters are provided,
            # perform RandomizedSearchCV
            if param_grid:

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=10,
                    scoring="r2",
                    cv=3,
                    random_state=42,
                    n_jobs=-1
                )

                random_search.fit(X_train, y_train)

                # Get the best tuned model
                best_model = random_search.best_estimator_

                # Replace original model with tuned model
                models[model_name] = best_model

            else:
                # No hyperparameters to tune
                model.fit(X_train, y_train)
                best_model = model

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # R2 scores
            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            print(model_name)
            print(
                "Train R2 Score:",
                round(train_model_score, 4)
            )
            print(
                "Test R2 Score:",
                round(test_model_score, 4)
            )
            print("-" * 50)

            # Store test score
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)