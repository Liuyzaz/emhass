#!/usr/bin/env python3

import logging
import time
import warnings

import numpy as np
import pandas as pd
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
# from pmdarima import auto_arima
from emhass import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MLForecaster:
    r"""
    A forecaster class using machine learning models with auto-regressive approach and features\
    based on timestamp information (hour, day, week, etc).

    This class uses the `skforecast` module and the machine learning models are from `scikit-learn`.

    It exposes three main methods:

    - `fit`: to train a model with the passed data.

    - `predict`: to obtain a forecast from a pre-trained model.

    - `tune`: to optimize the models hyperparameters using bayesian optimization.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_type: str,
        var_model: str,
        sklearn_model: str,
        num_lags: int,
        emhass_conf: dict,
        logger: logging.Logger,
    ) -> None:
        r"""Define constructor for the forecast class.

        :param data: The data that will be used for train/test
        :type data: pd.DataFrame
        :param model_type: A unique name defining this model and useful to identify \
            for what it will be used for.
        :type model_type: str
        :param var_model: The name of the sensor to retrieve data from Home Assistant. \
            Example: `sensor.power_load_no_var_loads`.
        :type var_model: str
        :param sklearn_model: The `scikit-learn` model that will be used. For now only \
            this options are possible: `LinearRegression`, `ElasticNet` and `KNeighborsRegressor`.\
            Advanced models: `LightGBM`, `SVR`, `ARIMA`
        :type sklearn_model: str
        :param num_lags: The number of auto-regression lags to consider. A good starting point \
            is to fix this as one day. For example if your time step is 30 minutes, then fix this \
            to 48, if the time step is 1 hour the fix this to 24 and so on.
        :type num_lags: int
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging.Logger
        """
        self.data = data
        self.model_type = model_type
        self.var_model = var_model
        self.sklearn_model = sklearn_model
        self.num_lags = num_lags
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.is_tuned = False
        # A quick data preparation
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    @staticmethod
    def neg_r2_score(y_true, y_pred):
        """The negative of the r2 score."""
        return -r2_score(y_true, y_pred)

    @staticmethod
    def generate_exog(data_last_window, periods, var_name):
        """Generate the exogenous data for future timestamps."""
        forecast_dates = pd.date_range(
            start=data_last_window.index[-1] + data_last_window.index.freq,
            periods=periods,
            freq=data_last_window.index.freq,
        )
        exog = pd.DataFrame({var_name: [np.nan] * periods}, index=forecast_dates)
        exog = utils.add_date_features(exog)
        return exog

    def fit(
        self,
        split_date_delta: str | None = "48h",
        perform_backtest: bool | None = False,
        # #added code for tuning days
        tuning_days: str | None = "5days",
        eval_metrics: bool | None = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""The fit method to train the ML model.

        :param split_date_delta: The delta from now to `split_date_delta` that will be used \
            as the test period to evaluate the model, defaults to '48h'
        :type split_date_delta: Optional[str], optional
        :param perform_backtest: If `True` then a back testing routine is performed to evaluate \
            the performance of the model on the complete train set, defaults to False
        :type perform_backtest: Optional[bool], optional
        :param tuning_days: Number of days to use for tuning, defaults to "5days"
        :type tuning_days: Optional[str], optional
        :param eval_metrics: If True, calculate and log additional evaluation metrics (RMSE, MAE, MAPE), defaults to True
        :type eval_metrics: Optional[bool], optional
        :return: The DataFrame containing the forecast data results without and with backtest
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        self.logger.info("Performing a forecast model fit for " + self.model_type)
        # Preparing the data: adding exogenous features
        self.data_exo = pd.DataFrame(index=self.data.index)
        self.data_exo = utils.add_date_features(self.data_exo)
        self.data_exo[self.var_model] = self.data[self.var_model]
        self.data_exo = self.data_exo.interpolate(method="linear", axis=0, limit=None)
        # train/test split
        # self.date_train = (
        #     self.data_exo.index[-1] - pd.Timedelta("5days") + self.data_exo.index.freq
        # )  # The last 5 days
        self.date_train = (
            self.data_exo.index[-1] - pd.Timedelta(tuning_days) + self.data_exo.index.freq
        )  
        self.date_split = (
            self.data_exo.index[-1]
            - pd.Timedelta(split_date_delta)
            + self.data_exo.index.freq
        )  # The last 48h
        self.data_train = self.data_exo.loc[
            : self.date_split - self.data_exo.index.freq, :
        ]
        self.data_test = self.data_exo.loc[self.date_split :, :]
        self.steps = len(self.data_test)
        # Pick correct sklearn model
        if self.sklearn_model == "LinearRegression":
            base_model = LinearRegression()
        elif self.sklearn_model == "ElasticNet":
            base_model = ElasticNet()
        elif self.sklearn_model == "KNeighborsRegressor":
            base_model = KNeighborsRegressor()
        # Add advanced models:
        elif self.sklearn_model == "LightGBM":
            base_model = lgb.LGBMRegressor(verbose=-1)
        elif self.sklearn_model == "SVR":
                # Create a pipeline with preprocessing steps
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Replace the basic SVR with a pipeline that includes scaling
            base_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))
            ])
        # elif self.sklearn_model == "ARIMA":
        #     # ARIMA is handled separately as it doesn't use ForecasterRecursive
        #     self._fit_arima_model()
        #     # Return early as ARIMA is handled differently
        #     return self._arima_results()
        else:
            self.logger.error(
                "Passed sklearn model "
                + self.sklearn_model
                + " is not valid. Defaulting to KNeighborsRegressor"
            )
            base_model = KNeighborsRegressor()
        # Define the forecaster object
        self.forecaster = ForecasterRecursive(regressor=base_model, lags=self.num_lags)
        # Fit and time it
        self.logger.info("Training a " + self.sklearn_model + " model")
        start_time = time.time()
        self.forecaster.fit(
            y=self.data_train[self.var_model],
            exog=self.data_train.drop(self.var_model, axis=1),
        )
        self.logger.info(f"Elapsed time for model fit: {time.time() - start_time}")
        # Make a prediction to print metrics
        predictions = self.forecaster.predict(
            steps=self.steps, exog=self.data_test.drop(self.var_model, axis=1)
        )
        pred_metric = r2_score(self.data_test[self.var_model], predictions)
        self.logger.info(
            f"Prediction R2 score of fitted model on test data: {pred_metric}"
        )
        # Add calculation of additional metrics if requested
        if eval_metrics:
            self._calculate_metrics(self.data_test[self.var_model], predictions)
        
        # Packing results in a DataFrame
        df_pred = pd.DataFrame(
            index=self.data_exo.index, columns=["train", "test", "pred"]
        )
        df_pred["train"] = self.data_train[self.var_model]
        df_pred["test"] = self.data_test[self.var_model]
        df_pred["pred"] = predictions
        df_pred_backtest = None
        if perform_backtest is True:
            # Using backtesting tool to evaluate the model
            self.logger.info("Performing simple backtesting of fitted model")
            start_time = time.time()
            cv = TimeSeriesFold(
                steps=self.num_lags,
                initial_train_size=None,
                fixed_train_size=False,
                gap=0,
                allow_incomplete_fold=True,
                refit=False,
            )
            metric, predictions_backtest = backtesting_forecaster(
                forecaster=self.forecaster,
                y=self.data_train[self.var_model],
                exog=self.data_train.drop(self.var_model, axis=1),
                cv=cv,
                metric=MLForecaster.neg_r2_score,
                verbose=False,
                show_progress=True,
            )
            self.logger.info(f"Elapsed backtesting time: {time.time() - start_time}")
            self.logger.info(f"Backtest R2 score: {-metric}")
            df_pred_backtest = pd.DataFrame(
                index=self.data_exo.index, columns=["train", "pred"]
            )
            df_pred_backtest["train"] = self.data_exo[self.var_model]
            df_pred_backtest["pred"] = predictions_backtest
        return df_pred, df_pred_backtest
    
    # def _fit_arima_model(self):
    #     """Fit ARIMA model using pmdarima's auto_arima."""
    #     self.logger.info("Training an ARIMA model")
    #     start_time = time.time()
        
    #     # Auto ARIMA will find the best order for an ARIMA model
    #     self.arima_model = auto_arima(
    #         self.data_train[self.var_model],
    #         exogenous=self.data_train.drop(self.var_model, axis=1),
    #         seasonal=True,
    #         m=24,  # Daily seasonality (adjust based on data frequency)
    #         suppress_warnings=True,
    #         error_action="ignore",
    #         trace=True if self.logger.level <= logging.INFO else False
    #     )
        
    #     self.logger.info(f"Elapsed time for ARIMA model fit: {time.time() - start_time}")
        
    #     # Make prediction for test period
    #     self.arima_predictions = self.arima_model.predict(
    #         n_periods=self.steps,
    #         exogenous=self.data_test.drop(self.var_model, axis=1)
    #     )
        
    #     # Convert to Series with datetime index
    #     self.arima_predictions = pd.Series(
    #         self.arima_predictions,
    #         index=self.data_test.index
    #     )
        
    #     # Calculate metrics
    #     pred_metric = r2_score(self.data_test[self.var_model], self.arima_predictions)
    #     self.logger.info(f"ARIMA model R2 score on test data: {pred_metric}")
        
    #     if hasattr(self, '_calculate_metrics'):
    #         self._calculate_metrics(self.data_test[self.var_model], self.arima_predictions)

    # def _arima_results(self):
    #     """Package ARIMA results into DataFrames similar to other models."""
    #     df_pred = pd.DataFrame(
    #         index=self.data_exo.index, columns=["train", "test", "pred"]
    #     )
    #     df_pred["train"] = self.data_train[self.var_model]
    #     df_pred["test"] = self.data_test[self.var_model]
    #     df_pred.loc[self.arima_predictions.index, "pred"] = self.arima_predictions
        
    #     return df_pred, None  # No backtest results for ARIMA

    def predict(self, data_last_window: pd.DataFrame | None = None) -> pd.Series:
        """The predict method to generate forecasts from a previously fitted ML model.

        :param data_last_window: The data that will be used to generate the new forecast, this \
            will be freshly retrieved from Home Assistant. This data is needed because the forecast \
            model is an auto-regressive model with lags. If not passed then the data used during the \
            model train is used, defaults to None
        :type data_last_window: Optional[pd.DataFrame], optional
        :return: A pandas series containing the generated forecasts.
        :rtype: pd.Series
        """
        if data_last_window is None:
            predictions = self.forecaster.predict(
                steps=self.num_lags, exog=self.data_test.drop(self.var_model, axis=1)
            )
        else:
            data_last_window = data_last_window.interpolate(
                method="linear", axis=0, limit=None
            )
            if self.is_tuned:
                exog = MLForecaster.generate_exog(
                    data_last_window, self.lags_opt, self.var_model
                )
                predictions = self.forecaster.predict(
                    steps=self.lags_opt,
                    last_window=data_last_window[self.var_model],
                    exog=exog.drop(self.var_model, axis=1),
                )
            else:
                exog = MLForecaster.generate_exog(
                    data_last_window, self.num_lags, self.var_model
                )
                predictions = self.forecaster.predict(
                    steps=self.num_lags,
                    last_window=data_last_window[self.var_model],
                    exog=exog.drop(self.var_model, axis=1),
                )
        return predictions

    def tune(self, debug: bool | None = False) -> pd.DataFrame:
        """Tuning a previously fitted model using bayesian optimization.

        :param debug: Set to True for testing and faster optimizations, defaults to False
        :type debug: Optional[bool], optional
        :return: The DataFrame with the forecasts using the optimized model.
        :rtype: pd.DataFrame
        """
        # Regressor hyperparameters search space
        if self.sklearn_model == "LinearRegression":
            if debug:

                def search_space(trial):
                    search_space = {
                        "fit_intercept": trial.suggest_categorical(
                            "fit_intercept", [True]
                        ),
                        "lags": trial.suggest_categorical("lags", [3]),
                    }
                    return search_space
            else:

                def search_space(trial):
                    search_space = {
                        "fit_intercept": trial.suggest_categorical(
                            "fit_intercept", [True, False]
                        ),
                        "lags": trial.suggest_categorical(
                            "lags", [6, 12, 24, 36, 48, 60, 72]
                        ),
                    }
                    return search_space
        elif self.sklearn_model == "ElasticNet":
            if debug:

                def search_space(trial):
                    search_space = {
                        "selection": trial.suggest_categorical("selection", ["random"]),
                        "lags": trial.suggest_categorical("lags", [3]),
                    }
                    return search_space
            else:

                def search_space(trial):
                    search_space = {
                        "alpha": trial.suggest_float("alpha", 0.0, 2.0),
                        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                        "selection": trial.suggest_categorical(
                            "selection", ["cyclic", "random"]
                        ),
                        "lags": trial.suggest_categorical(
                            "lags", [6, 12, 24, 36, 48, 60, 72]
                        ),
                    }
                    return search_space
        elif self.sklearn_model == "KNeighborsRegressor":
            if debug:

                def search_space(trial):
                    search_space = {
                        "weights": trial.suggest_categorical("weights", ["uniform"]),
                        "lags": trial.suggest_categorical("lags", [3]),
                    }
                    return search_space
            else:

                def search_space(trial):
                    search_space = {
                        "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
                        "leaf_size": trial.suggest_int("leaf_size", 20, 40),
                        "weights": trial.suggest_categorical(
                            "weights", ["uniform", "distance"]
                        ),
                        "lags": trial.suggest_categorical(
                            "lags", [6, 12, 24, 36, 48, 60, 72]
                        ),
                    }
                    return search_space
            # Add new hyperparameter search spaces
        elif self.sklearn_model == "LightGBM":
            if debug:
                def search_space(trial):
                    search_space = {
                        "n_estimators": trial.suggest_categorical("n_estimators", [100]),
                        "lags": trial.suggest_categorical("lags", [3]),
                    }
                    return search_space
            else:
                def search_space(trial):
                    search_space = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                        "lags": trial.suggest_categorical("lags", [6, 12, 24, 36, 48, 60, 72]),
                    }
                    return search_space
        elif self.sklearn_model == "SVR":
            if debug:
                def search_space(trial):
                    search_space = {
                        "C": trial.suggest_categorical("C", [1.0]),
                        "lags": trial.suggest_categorical("lags", [3]),
                    }
                    return search_space
            else:
                def search_space(trial):
                    search_space = {
                        "C": trial.suggest_float("C", 0.1, 10.0, log=True),
                        "gamma": trial.suggest_float("gamma", 0.001, 1.0, log=True),
                        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
                        "lags": trial.suggest_categorical("lags", [6, 12, 24, 36, 48, 60, 72]),
                    }
                    return search_space

        # Bayesian search hyperparameter and lags with skforecast/optuna
        # Lags used as predictors
        if debug:
            refit = False
            num_lags = 3
        else:
            refit = True
            num_lags = self.num_lags
        # The optimization routine call
        self.logger.info("Bayesian hyperparameter optimization with backtesting")
        start_time = time.time()
        cv = TimeSeriesFold(
            steps=num_lags,
            initial_train_size=len(self.data_exo.loc[: self.date_train]),
            fixed_train_size=True,
            gap=0,
            skip_folds=None,
            allow_incomplete_fold=True,
            refit=refit,
        )
        self.optimize_results, self.optimize_results_object = (
            bayesian_search_forecaster(
                forecaster=self.forecaster,
                y=self.data_train[self.var_model],
                exog=self.data_train.drop(self.var_model, axis=1),
                cv=cv,
                search_space=search_space,
                metric=MLForecaster.neg_r2_score,
                n_trials=30, #increase this for more trials
                random_state=123,
                return_best=True,
            )
        )
        self.logger.info(f"Elapsed time: {time.time() - start_time}")
        self.is_tuned = True
        predictions_opt = self.forecaster.predict(
            steps=self.num_lags, exog=self.data_test.drop(self.var_model, axis=1)
        )
        freq_hours = self.data_exo.index.freq.delta.seconds / 3600
        self.lags_opt = int(np.round(len(self.optimize_results.iloc[0]["lags"])))
        self.days_needed = int(np.round(self.lags_opt * freq_hours / 24))
        df_pred_opt = pd.DataFrame(
            index=self.data_exo.index, columns=["train", "test", "pred_optim"]
        )
        df_pred_opt["train"] = self.data_train[self.var_model]
        df_pred_opt["test"] = self.data_test[self.var_model]
        df_pred_opt["pred_optim"] = predictions_opt
        pred_optim_metric_train = -self.optimize_results.iloc[0]["neg_r2_score"]
        self.logger.info(
            f"R2 score for optimized prediction in train period: {pred_optim_metric_train}"
        )
        pred_optim_metric_test = r2_score(
            df_pred_opt.loc[predictions_opt.index, "test"],
            df_pred_opt.loc[predictions_opt.index, "pred_optim"],
        )
        self.logger.info(
            f"R2 score for optimized prediction in test period: {pred_optim_metric_test}"
        )
        self.logger.info("Number of optimal lags obtained: " + str(self.lags_opt))
        return df_pred_opt

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate and log multiple evaluation metrics."""
        # R² score - already used in main code
        r2 = r2_score(y_true, y_pred)
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        # Mean Absolute Percentage Error - with handling for zeros
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        self.logger.info(f"Model Evaluation Metrics:")
        self.logger.info(f"R² Score: {r2:.4f}")
        self.logger.info(f"RMSE: {rmse:.4f}")
        self.logger.info(f"MAE: {mae:.4f}")
        self.logger.info(f"MAPE: {mape:.2f}%")
        
        # Store metrics for later access
        self.metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        return self.metrics