import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error

from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score, train_test_split

from eda_helpers import count_outliers


def custom_corr(data: pd.DataFrame, data_info: pd.DataFrame, features: list) -> pd.DataFrame:
    """ Calculate the correlations between all possible pairs of numerical data features depending on their distributions.

    Args:
        data: pd.DataFrame
            dataframe to pick the features from
        data_info: pd.DataFrame
            dataframe with information about distribution of data features
        features: list
            list of features between which correlation analysis will be applied

    Returns:
        summary: pd.DataFrame: 
            dataframe with the following information about correlations: 
             - method: Pearson or Spearman
             - feature1 and feature2
             - r-value: correlation coefficient
             - p-value: how signifficant is that correlation. 
             - stat-sign: bool. Significance theshold is p-value = 0.05, so 0.04 is significant, 0.06 - not.
             - N: number of observations in each feature.
        r-values: pd.DataFrame
            correlation matrix (dataframe) of n features by n features size
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



def show_outliers_importance (data: pd.DataFrame, data_info: pd.DataFrame, target_feature: str, corr_feature_list: list):
    """Calculates delta of correlations between features with and without outliers. Prots the results as a heatmap.

    Args:
        data: pd.DataFrame
            dataframe to pick the features from
        data_info: pd.DataFrame
            dataframe with information about distribution of data features
        target_feature: str
            name of target feature, from which the outliers shoud be removed
        corr_feature_list: list 
            list of names of features, that will be checked for correlation with target
    """

    # 1. identify the outliers and their borders
    feature_outliers = count_outliers(data = data, data_info = data_info, features = [target_feature])
    lower_threshold = feature_outliers.loc[target_feature, 'lower_threshold']
    upper_threshold = feature_outliers.loc[target_feature, 'upper_threshold']

    corr_feature_list.append(target_feature)
    
    # 2. create a corr matrix with features of interest and target with outliers
    _, corr_with_outliers  = custom_corr(data, data_info, corr_feature_list)

    # 3. remove all outliers and leverage points of feature
    no_outliers_df = data.copy()
    if lower_threshold is not np.NaN:
        no_outliers_df.drop(no_outliers_df.loc[no_outliers_df[target_feature]<lower_threshold].index, inplace = True)
    if upper_threshold is not np.NaN:
        no_outliers_df.drop(no_outliers_df.loc[no_outliers_df[target_feature]>upper_threshold].index, inplace = True)

    # 4. create new corr matrix
    _, corr_without_outliers = custom_corr(no_outliers_df, data_info, corr_feature_list)

    # 5. calculate the delta of 2 matrixes to define if the outliers are influential
    delta_corr = corr_without_outliers- corr_with_outliers

    # 6. plot delta correlations
    mask=np.triu(np.ones_like(delta_corr, dtype=bool))
    heatmap = sns.heatmap(delta_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')



def custom_anova(data: pd.DataFrame, grouping_var: list, feature: str, result_table: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """ Runs anova analysis for a list of nominal features, grouped by another feature

    Args:
        data: pd.DataFrame
            dataframe to pick the features from
        grouping_var: list
            list of feature, on which anova will be conducted
        feature: str
            variable to group features by
        result_table: pd.DataFrame
            dataFrame to put the results in
        plot: bool, optional
            plot the results as a boxplot. Defaults to True.

    Returns:
        result_table: pd.DataFrame
            table with the folloving data:
            - test-type: 'One way ANOVA' or 'Welch ANOVA'
            - feature: name of the feature, on which the anova was conducted
            - group-var: different feature, by which the previous one was groupped (always the same)
            - f-value: the ratio of explained variance to unexplained variance
            - p-value: determines if the difference between group means is statistically significant
            - stat-sign: True, if p-value is less than 0.05. Otherwise False
            - variances: equal or no_equal. Determined by levene test.
    """
    for col in grouping_var:

        # check is variances are homogeneous
        values_per_group = {
            grp_label: values
            for grp_label, values in data.groupby(col, observed=True)['Price']
        }
        
        # create a list with lists of values
        (_, levene_p_value) = stats.levene(*values_per_group.values())
        if levene_p_value >0.05:
            variances = 'equal'
        else:
            variances = 'not_equal'

        # normal ANOVA can be applied only when the variances are homogeneous
        if variances == 'equal':
            test_type = 'One way ANOVA'
            (f_value, p_value) = stats.f_oneway(*values_per_group.values())
        else:
            test_type = 'Welch ANOVA'
            welch_df = pg.welch_anova(dv = feature, between = col, data = data)
            p_value = welch_df['p-unc']
            f_value = welch_df['F']

        dict_result = {
            "test-type": test_type,
            "feature": col,
            "group-var": feature,
            "f-value": round(f_value, 3),
            "p-value": round(p_value, 5),
            "stat-sign": (p_value < 0.05),
            "variances": variances
        }
        df_result = pd.DataFrame(data=dict_result, index=[0])
        result_table = pd.concat([result_table, df_result], ignore_index=True) 

        # plot grouping var vs feature as boxplots
        if plot:
            _, ax = plt.subplots(figsize = (15,4))
            _ = sns.boxplot(ax=ax, x=data[col], y=data[feature])
            _ = sns.swarmplot(
                ax=ax, x=data[col], y= data[feature], color=".25", alpha=0.50, size=2
            )
            _ = ax.set_title(f"Boxplot {feature} across {col}")
            plt.xticks(rotation=90)

    return result_table



def evaluate_model(model, X: pd.DataFrame, y: np.array, results: pd.DataFrame, cv: int = 5) :
    """Calculate RMSE, MAE, explained variation and correlation coeficient of predicted values for train and test sets, and add the results to the 'results' dataframe
    Args:
        model:
            sklearn object, e.g. LogisticRegression, LinearRegression, etc.
        X_train, X_valid: pd.DataFrame
            pd.DataFrame of features, used in model (X)
        y_train, y_valid: np.array
            array of target values
        results: pd.DataFrame
            table, where the row with evaluations will be added
        cv: int
            number of folds for cross-validation. Deafualts to 5.

    Returns:
        RMSE: float
            root mean squared error. The less, the better
        MAE: float
            mean absolute error. The less, the better
        r2_coef_determination: float
            how well a statistical model predicts an outcome. The closer to 1, the better 
        r-explained_variance: float
            proportion of explained variance. The closer to 1, the better 
        corr: float
            correlation between real and predicted value. The closer to 1, the better 
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    r2_train = cross_val_score(model, X_train, y_train, cv=cv, scoring = 'r2')
    r2_train = r2_train.mean()

    explained_variance_train = cross_val_score(model, X_train, y_train, cv=cv, scoring = 'explained_variance')
    explained_variance_train = explained_variance_train.mean()

    rmses_train = []
    maes_train = []
    corrs_train = []

    for _ in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1/cv)
        model.fit(X_train, y_train)
        y_pred1 = model.predict(X_test)

        # back transformation from log of price to price
        y_true1 = np.exp(y_test)
        y_pred1 = np.exp(y_pred1)

        RMSE = mean_squared_error(y_true1, y_pred1, squared=False)
        MAE = mean_absolute_error(y_true1, y_pred1)
        corr = stats.spearmanr(y_true1, y_pred1)[0]

        rmses_train.append(RMSE)
        maes_train.append(MAE)
        corrs_train.append(corr)

    RMSE_train = int(np.mean(rmses_train))
    MAE_train = int(np.mean(maes_train))
    corr_train = round(np.mean(corrs_train), 4)
    vifs = [round(variance_inflation_factor(X_train.values, i), 2) for i in range(len(X_train.columns))]
    
    # perform all measures on validation set

    # back transformation from log of price to price
    y_valid = np.exp(y_valid)
    y_valid_pred = np.exp(y_valid_pred)

    r2_valid = r2_score(y_valid, y_valid_pred)
    explained_variance_valid = explained_variance_score(y_valid, y_valid_pred)
    RMSE_valid = int(mean_squared_error(y_valid, y_valid_pred, squared=False))
    MAE_valid = int(mean_absolute_error(y_valid, y_valid_pred))
    corr_vlid = stats.spearmanr(y_valid, y_valid_pred)[0]

    new_row = [model, list(X_train.columns), RMSE_train, RMSE_valid, MAE_train, MAE_valid, r2_train, r2_valid, explained_variance_train, explained_variance_valid, corr_train, corr_vlid, vifs]
    results.loc[len(results)] = new_row

