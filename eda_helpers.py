import pandas as pd
import numpy as np
import scipy.stats as stats


def normality_check(features: list, data:pd.DataFrame, data_info:pd.DataFrame) -> pd.DataFrame:
    """ Define if the numerical feature is distributed normally

    Args:
        data: pd.DataFrame
            dataset, in which features distributions have to be checked for normality
        data_info: pd.DataFrame
            dataframe, where the values will be stored

    Returns:
        pd.DataFrame
            data_info dataframe with added values to 'Distribution' feature
    """

    list_numerical = []
    for i in features:
        if data_info.loc[i, 'data_type'] == "continuous" or data_info.loc[i, 'data_type'] == "descrete":
            list_numerical.append(i)

    p_values_normality = data[list_numerical].apply(
            lambda x: stats.kstest(
                x.dropna(),
                stats.norm.cdf,
                args=(np.nanmean(x), np.nanstd(x)),
                N=len(x),
            )[1],
            axis=0
    )
    
    for feature in list_numerical:
        
        if p_values_normality[feature] > 0.05:
            data_info.loc[feature, 'distribution'] = 'normal'
        else: 
            data_info.loc[feature, 'distribution'] = 'not normal'

    return data_info



def count_outliers(data:pd.DataFrame, data_info:pd.DataFrame, features: list, show_details: bool = False):
    """ Calculation and display (optional) of outliers of selected features according to type of distribution

    Args:
        data: pd.DataFrame
            main dataset, from which the data is taken to define outliers
        data_info: pd.DataFrame
            dataset, which stores info about type of distribution in 'distribution' column
        features: list
            list of features where outliers have to be defined and calculated
        show_details: bool, optional. Defaults to False.
            print out the table with only those rows, that were marked as outliers
    """
    for i in features:
        
        distribution = data_info.loc[i, 'distribution']
        
        if distribution == 'right_skewed':
            Q3 = np.nanpercentile(data[i], [75])[0]
            IQR = stats.iqr(data[i], interpolation = 'midpoint', nan_policy='omit')

            outlier_border = Q3 + 1.5*IQR

            outliers = data[data[i]>outlier_border]
            print(f'o {outliers.shape[0]} datapoints with {i} > {outlier_border}')

            if show_details:
                display(outliers)

        elif distribution == 'left_skewed':
            for i in data.columns:
                Q1 = np.nanpercentile(data[i], [25])[0]
                IQR = stats.iqr(data[i], interpolation = 'midpoint', nan_policy='omit')

                outlier_border = Q1 - 1.5*IQR

                outliers = data[data[i]>outlier_border]
                print(f'o {outliers.shape[0]} datapoints with {i} > {outlier_border}')

                if show_details:
                    display(outliers)

        elif (distribution == 'normal') or (distribution == 'heavy_tailed'):
            
            mean = data[i].mean()
            std = data[i].std()
            lower_treshold = mean - 3 * std
            upper_threshold = mean + 3 * std
            values_below_3std = data.loc[data[i] < lower_treshold, i].values
            values_above_3std = data.loc[data[i] > upper_threshold, i].values
            if values_below_3std.shape[0] >0:
                print(f'o {values_below_3std.shape[0]} datapoints with {i} < {lower_treshold}')
            if values_above_3std.shape[0]>0:
                print(f'o {values_above_3std.shape[0]} datapoints with {i} > {upper_threshold}')

        
            outliers_low = data[data[i]<lower_treshold]
            outliers_up = data[data[i]>upper_threshold]
            outliers = pd.concat((outliers_low, outliers_up))

            if show_details:
                display(outliers)
                
        else: 
            print(f'Impossible to define outliers for {i} data')