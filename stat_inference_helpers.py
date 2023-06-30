import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error
from itertools import combinations

from sklearn.model_selection import cross_val_score, train_test_split


def custom_corr(data: pd.DataFrame, data_info: pd.DataFrame, features: list) -> pd.DataFrame:
    """ Calculate the correlations between all possible pairs of numerical data features depending on their distributions.

    Args:
        data: pd.DataFrame
            dataframe with features
        data_info: pd.DataFrame
            dataframe with information about distribution of data features

    Returns:
        summary: pd.DataFrame: 
            dataframe with the following information about correlations: 
             - method: Pearson or Spearman
             - feature1 and feature2
             - r-value: correlation coefficient
             - p-value: how signifficant is that correlation. 
             - stat-sign: bool. Significance theshold is p-value = 0.05, so 0.04 is significant, 0.06 - not.
             - N: number of observations in each feature.
    """
    r_values = pd.DataFrame(1, columns=features, index=features)
    summary = pd.DataFrame()
    
    # create lists with normaly / not normaly distributed features
    norm_features = []
    no_norm_features = []
    for i in features:
        if (data_info.loc[i, 'data_type'] == 'continuous') or (data_info.loc[i, 'data_type'] == 'descrete') :
            if data_info.loc[i, 'distribution'] == 'normal':
                norm_features.append(i)
            else:
                no_norm_features.append(i)

    # create list of all possible combinations of features without repeats
    iterator = combinations(norm_features+no_norm_features, 2)

    # get correlations between eve of features and it's signifficance
    for col1, col2 in iterator:
        if col1 in norm_features and col2 in norm_features:
            r_value, p_value = stats.pearsonr(data.loc[:, col1], data.loc[:, col2])
            method = 'Pearson'
            r_values.loc[col1, col2] = r_value
            r_values.loc[col2, col1] = r_value
        else: 
            r_value, p_value = stats.spearmanr(data.loc[:, col1], data.loc[:, col2])
            method = 'Spearman'
            r_values.loc[col1, col2] = r_value
            r_values.loc[col2, col1] = r_value
        n = len(data)

        # Store output in dataframe format
        dict_summary = {
            "method": method,
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n,
        }
        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )
    return summary, r_values



def evaluate_model(model, features: pd.DataFrame, target: np.array, results: pd.DataFrame, cv: int = 5) :
    """Calculate RMSE, MAE, explained variation and correlation coeficient of predicted values and add the results to the 'results' dataframe
    Args:
        model_type: str
            e.g. Logostic regression, Linear regression, etc.
        X_columns: list
            list of features, used in model
        target_name: str
            name of predicted variable, that was used in model, e.g. log(Price) or Price
        y_true: list
            list of true target values
        y_pred: list. 
            list of predicted target values
        results: pd.DataFrame
            table, where the row with evaluations will be added

    Returns:
        RMSE: float
            root mean squared error. The less, the better
        MAE: float
            mean absolute error. The less, the better
        r-value: float
            proportion of explained variance. The closer to 1, the better 
        corr: float
            correlation between real and predicted value. The closer to 1, the better 
    """
    
    model.fit(features, target)
    r2_coef_determination = cross_val_score(model, features, target, cv=cv, scoring = 'r2')
    r2_coef_determination = r2_coef_determination.mean()

    explained_variance = cross_val_score(model, features, target, cv=cv, scoring = 'explained_variance')
    explained_variance = explained_variance.mean()

    rmses = []
    maes = []
    corrs = []
    for _ in range(cv):

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=1/cv)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # back transformation from log of price to price
        y_true = np.exp(y_test)
        y_pred = np.exp(y_pred)

        RMSE = mean_squared_error(y_true, y_pred, squared=False)
        MAE = mean_absolute_error(y_true, y_pred)
        corr = stats.spearmanr(y_true, y_pred)[0]

        rmses.append(RMSE)
        maes.append(MAE)
        corrs.append(corr)

    RMSE = int(np.mean(rmses))
    MAE = int(np.mean(maes))
    corr = round(np.mean(corrs), 4)

    new_row = [model, cv, list(features.columns), RMSE, MAE, r2_coef_determination, explained_variance, corr]
    results.loc[len(results)] = new_row

    return RMSE, MAE, r2_coef_determination, explained_variance, corr