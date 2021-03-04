'''
In the following code snippet, I've created a machine learning (ML) model evaluation framework that can be
used to evaluate the performance of five different ML models for any target variable (y) and
for a set of predictor variables (X) that are passed as dataframes.
While evaluating the performance of each model, a hyperparameter tuning to select optimal parameters
is performed in order to have the best fit for each ML model.

The code below is very generalizable since :
1. It can evaluate five common ML models
2. The code can be easily extended to support additional ML models
3. It can accept any dataframe of predictor variables (X) and target variable (y)

'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet


class BaseModelEvaluator:
    def __init__(self, model_type, model_gridsearch, pca_apply=False):
        """
        Args:
            model_type: str
                "RF","Linear","ElasticNet", Ridge","XGBoost"
            model_gridsearch: sklearn.model_selection.GridSearchCV
                Stores all the combinations of hyperparameter grid values for the model
            pca_apply: bool
                Whether or not principal component analysis will be applied to the dataframe (X) while evaluating the
                model
        """
        self.model_type = model_type
        self.model_gridsearch = model_gridsearch
        self.pca_apply = pca_apply

    ###########################################################
    ### Create a a function to evalulate model performance ####
    ###########################################################

    def calculate_result(self, X_train, X_test, y_train, y_test):
        """
        Args:
            X_train: pandas.DataFrame
                Predictor variable for training model
            X_test: pandas.DataFrame
                Predictor variable for testing model
            y_train: pandas.DataFrame
                Target variable for training model
            y_test: pandas.DataFrame
                Target variable for testing model

        Returns:
            object: EvaluatorResult """

        # Perform a model fit with training data
        self.model_gridsearch.fit(X_train, y_train)

        # Make predictions on the test data based on  model fit
        y_pred = []
        y_pred = self.model_gridsearch.predict(X_test)

        # Linear regression model returns nparray of nparray objects; convert to nparray
        if self.model_type == "Linear":
            y_pred = [y[0] for y in y_pred]

        # Determine best parameters using best_params_ attribute
        best_parameters = self.model_gridsearch.best_params_
        print("Best parameter for {} is {}".format(self.model_type, best_parameters))

        # Determine best accuracy using best_score_ attribute
        best_result = self.model_gridsearch.best_score_
        print("Best accuracy achieved for {} is {}".format(self.model_type, best_result))

        # Calculate test and train R2
        train_score = self.model_gridsearch.score(X_train, y_train)
        test_score = self.model_gridsearch.score(X_test, y_test)
        print("Train R2 = {}, Test_R2 = {}".format(train_score, test_score))

        # Calculate root mean squared error for predicted and actual values
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error: {}".format(rmse))

        model_result = EvaluatorResult(y_pred, best_parameters, best_result, test_score, rmse)

        return model_result

        ####################################################################
        ### Create a graph of predicted vs actual values for each model ####
        ####################################################################

    def plot_results(self, model_result, y_test):
        """
        Args:
            model_result:
            y_test: pandas.DataFrame
                Target variable for testing model
        Returns:
            plot object
        """
        # Instantiate a plot object and set axis = ax
        plt.figure(figsize=(8, 6))

        # Use snsregplot to plot predicted and actual values of target variable
        ax = sns.regplot(x=y_test, y=model_result.y_pred, fit_reg=True)

        # Set the plot X and Y labels to any desired values. Here I'm setting them to be NO2 concentrations.
        ax.set(xlabel='Actual NO2 Concentrations (ppb)', ylabel='Predicted NO2 Concentrations (ppb)')

        # Add Plot title. Update the plot title to match your dataset
        plt.title("Predicted vs. Actual NO2 concentration for " + self.model_type)

        # Add the test R2 value to the plot. Here I'm spacing the text at (45,10) but this value can be edited
        plt.text(45, 10, "R2 = {}".format(round(model_result.test_score, 3)), fontsize=18)


class EvaluatorResult:
    '''
       This class
    '''
    def __init__(self, y_pred, best_parameters, best_result, test_score, rmse):
        self.y_pred = y_pred
        self.best_parameters = best_parameters
        self.best_result = best_result
        self.test_score = test_score
        self.rmse = rmse


'''Next, a subclass for each model that inherits from the parent class "BaseModelEvaluator" is created. Each subclass
contains the hyperparameter grid values, returns a gridsearchCV (grid search cross validation) object. '''


# Define random forest GridsearchCV subclass
class RandomForestGridSearchCV(BaseModelEvaluator):
    def __init__(self):

        # Instantiate a random forest regression model and pass input parameters such as random state, max_features
        #stading point for number of estimators
        rf_regressor = RandomForestRegressor(random_state=0, max_features='sqrt', n_estimators=50)

        # Random forest accepts 'number of estimators' and 'max depth' as grid parameters
        grid_param = {
            'n_estimators': [300, 400, 500, 600, 800],
            'max_depth': [5, 10, 15, 50]
        }

        # Instantiate a random forest model, perform grid search across all grid parameters using a 5-fold
        # cross validation. Grid search is optimized for best R2 scoring value.
        model_gridsearch = GridSearchCV(
            estimator=rf_regressor,
            param_grid=grid_param,
            scoring='r2',
            cv=5,
            n_jobs=-1)

        super().__init__('RF', model_gridsearch)


# Define Ridge regression GridsearchCV subclass
class RidgeGridSearchCV(BaseModelEvaluator):

    # Here pca_apply = True performs a principal component analysis on the dataset
    def __init__(self, pca_apply):
        """

        Args:
            pca_apply: bool
                Whether or not principal component analysis will be applied to the dataframe (X) while evaluating the
                model
        """
        # Instantiate a ridge regression model
        ridge_reg = Ridge(normalize=True)

        # Ridge accepts 'alpha (Regularization strength)' and 'fit_intercept' as grid parameters
        grid_param = {
            'alpha': [1, 0.1, 0.01, 0.001, 0, 10],
            'fit_intercept': [True, False],
        }

        # Instantiate a ridge regression model, perform grid search across all grid parameters using a 5-fold
        # cross validation. Grid search is optimized for best R2 scoring value.
        model_gridsearch = GridSearchCV(
            estimator=ridge_reg,
            param_grid=grid_param,
            scoring='r2',
            cv=5)

        super().__init__('Ridge', model_gridsearch, pca_apply)

    # Here we override the calculate_result method of the parent class inside the subclass since
    # calculate_result method for ridge regression should be evaluated differently

    def calculate_result(self, X_train, X_test, y_train, y_test):
        """
          Args:
            X_train: pandas.DataFrame
                Predictor variable for training model
            X_test: pandas.DataFrame
                Predictor variable for testing model
            y_train: pandas.DataFrame
                Target variable for training model
            y_test: pandas.DataFrame
                Target variable for testing model

        Returns:
            object: EvaluatorResult
        """

        # Copy X_train and X_test into separate variables
        X_train_copy = X_train
        X_test_copy = X_test

        # If pca_apply = True, apply PCA on X_train and X_test and store transformed dataframe in X_train_copy
        # and X_test_copy
        if self.pca_apply:
            pca = PCA(n_components=np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.999)[0][0])
            X_train_copy = pca.fit_transform(X_train)
            X_test_copy = pca.transform(X_test)

        # Perform a model fit with training data
        self.model_gridsearch.fit(X_train_copy, y_train)

        # Make predictions on the test data based on the model fit
        y_pred = []
        y_pred = self.model_gridsearch.predict(X_test_copy)

        # Ridge regression model returns nparray of nparray objects; convert to nparray
        y_pred = [y[0] for y in y_pred]

        # Calculate test and train R2
        train_score = self.model_gridsearch.score(X_train_copy, y_train)
        test_score = self.model_gridsearch.score(X_test_copy, y_test)
        print("Train R2 = {}, Test_R2 = {}".format(train_score, test_score))

        # Determine best parameters using best_params_ attribute
        best_parameters = self.model_gridsearch.best_params_
        print("Best parameter for {} is {}".format(self.model_type, best_parameters))

        # Determine best accuracy using best_score_ attribute
        best_result = self.model_gridsearch.best_score_
        print("Best R2 achieved for {} is {}".format(self.model_type, best_result))


# Define Linear regression GridsearchCV subclass
class LinearGridSearchCV(BaseModelEvaluator):
    def __init__(self):
        # Instantiate linear regression model
        linear_reg = LinearRegression()

        # Ridge accepts 'fit_intercept' a grid parameter
        grid_param = {'fit_intercept': [True, False]}

        # Instantiate a linear regression model, perform grid search across all grid parameters using a 5-fold
        # cross validation. Grid search is optimized for best R2 scoring value.
        model_gridsearch = GridSearchCV(estimator=linear_reg, param_grid=grid_param,
                                        cv=5, n_jobs=-1)

        super().__init__('Linear', model_gridsearch)


# Define ElasticNet regression GridsearchCV subclass
class ElasticNetGridSearchCV(BaseModelEvaluator):
    def __init__(self):
        # Instantiate elasticnet regression model and specify random state
        elastic_net = ElasticNet(normalize=True, random_state=0)

        # ElasticNet accepts 'l1_ratio' and 'alpha' as grid parameters
        grid_param = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
                      'alpha': [1, 0.1, 0.01, 0, 10]}

        # Instantiate an elasticnet model, perform grid search across all grid parameters using a 5-fold
        # cross validation. Grid search is optimized for best R2 scoring value.
        model_gridsearch = GridSearchCV(estimator=elastic_net,
                                        param_grid=grid_param,
                                        scoring='r2',
                                        cv=5)

        super().__init__('ElasticNet', model_gridsearch)


# Define XGBoost  GridsearchCV subclass
class XGBGridSearchCV(BaseModelEvaluator):
    def __init__(self):
        # Instantiate XGBmodel and specify random state and objective function
        XGB_reg = XGBRegressor(objective='reg:linear', random_state=0, n_estimators=50)

        # XGBoost accepts 'max_depth' , 'n_estimators' and 'learning_rate' as grid parameters
        grid_param = {
            'max_depth': [2, 4, 6, 10],
            'n_estimators': [40, 60, 100],
            'learning_rate': [0.1, 0.001, 0.05, 0.01]
        }

        # Instantiates XGB and perform a grid search across all grid parameters using a
        # 5-fold cross validation. The grid search is optimized for the best R2 scoring value.
        model_gridsearch = GridSearchCV(estimator=XGB_reg,
                                        param_grid=grid_param,
                                        cv=5)

        super().__init__('XGB', model_gridsearch)


def cv_gridsearch_models(model_type, X, y, pca_apply=False):
    """
    Args:
        model_type: str
            "RF","Linear","ElasticNet", Ridge","XGBoost"
        X: pandas.DataFrame
            Predictor variables or set of features
        y: pandas.DataFrame
            Target variable
        pca_apply: bool
                Whether or not principal component analysis will be applied to the dataframe (X) while evaluating the
                model

    Returns:

    """
    # Split X and y into 70/30 train and test set using train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Scale the test and training X - data using StandardScaler method
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Create a dictionary that maps the model type with the subclass object
    model_type_evaluator_map = {
        'RF': RandomForestGridSearchCV(),
        'Ridge': RidgeGridSearchCV(pca_apply),
        'Linear': LinearGridSearchCV(),
        'ElasticNet': ElasticNetGridSearchCV(),
        'XGB': XGBGridSearchCV()
    }

    # Calculate and plot results based on model type
    evaluator = model_type_evaluator_map[model_type]
    model_result = evaluator.calculate_result(X_train, X_test, y_train, y_test)
    evaluator.plot_results(model_result, y_test)





