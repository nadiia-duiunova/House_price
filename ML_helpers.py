import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import scipy.sparse as sps



def preprocess_data(data: pd.DataFrame, 
                    TARGET: str,
                    numerical_features_list: list, categorical_features_list: list,  ordinal_feature: str = '', order_of_categories: list = []
                    ) -> pd.DataFrame:
    """Transform the data according to it's format in order to feed it to the model.
    
    Parameters
    ----------
        data : pdandas.DataFrame 
            Dataframe with variables in columns and instances in rows, where data is represented in original data types.
        TARGET : str
            Name of target variable
        numerical_features_list : list
            List of features, that have numerical format in original dataframe
        categorical_features_list : list
            List of features, that are represented as categories in original dataframe
        ordinal_feature: str
            This function can precess only 1 ordinal feature, will be optimized in future
        order_of_categories: list
            Here you have to provide the right ascending order of values of the ordinal feature as a list 
        
    Returns
    -------
        preprocessed_data : pandas.DataFrame
            Preprocessed data, ready to be fed to the model
    """

    X = data.drop(columns=[TARGET])
    y = list(data[TARGET])

    if ordinal_feature != '':
        if not order_of_categories:
            raise ValueError('order_of_categories cannot be empty')
        if len(order_of_categories) != len(data[ordinal_feature].unique()):
            raise ValueError('incorrect number of categories in order_of_categories')
        if numerical_features_list:
            if categorical_features_list:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('stand scaler', StandardScaler(), numerical_features_list),
                    ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                    remainder='drop')
            else:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('stand scaler', StandardScaler(), numerical_features_list)],
                    remainder='drop')
        else:
            if categorical_features_list:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                    remainder='drop')
            else:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature))],
                    remainder='drop')
    elif numerical_features_list:
        if categorical_features_list:
            columntransformer = ColumnTransformer(transformers = [
                ('stand scaler', StandardScaler(), numerical_features_list),
                ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                remainder='drop')
        else:
            columntransformer = ColumnTransformer(transformers = [
            ('stand scaler', StandardScaler(), numerical_features_list)],
            remainder='drop')
    else:
        columntransformer = ColumnTransformer(transformers = [
            ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
            remainder='drop')


    X_trans = columntransformer.fit_transform(X)

    if sps.issparse(X_trans):
        X_trans = X_trans.toarray()

    x_columns_names = columntransformer.get_feature_names_out()
    X_trans = pd.DataFrame(X_trans, columns = x_columns_names)

    y_trans = pd.DataFrame(data = y, index=range(0, len(y)), columns=[TARGET])

    # for categorical target create a dictionary with substituting every category with a number and apply to target
    if all(isinstance(n, str) for n in y_trans[TARGET]):
        n_unique = len(y_trans[TARGET].unique())
        dict_of_values = {}
        for i in range(n_unique):
            key = y_trans[TARGET].unique()[i]
            dict_of_values[key] = i
            
        y_trans[TARGET] = y_trans[TARGET].replace(dict_of_values)

    # for numerical target - apply StandardScaler()
    else: 
        scaler = StandardScaler()
        y_trans[TARGET] = scaler.fit_transform(y_trans)

    preprocessed_data = pd.merge(left=y_trans, right=X_trans, left_index=True, right_index=True)


    return preprocessed_data