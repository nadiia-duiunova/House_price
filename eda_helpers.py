import pandas as pd
import numpy as np
import scipy.stats as stats


def normality_check(data:pd.DataFrame, data_info:pd.DataFrame, features: list,) -> pd.DataFrame:
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



def count_outliers(data:pd.DataFrame, data_info:pd.DataFrame, features: list, show_details: bool = True) -> pd.DataFrame:
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

    Returns:
        outliers_info: pd.DataFrame
            dataframe with 
    """
    outliers_info = pd.DataFrame(columns=['lower_threshold', 'upper_threshold', 'n_outliers'], index=features)

    for feature in features:
        if data_info.loc[feature, 'data_type'] in ['continuous', 'descrete']:
            distribution = data_info.loc[feature, 'distribution']
            
            if distribution == 'right_skewed':
                Q3 = np.nanpercentile(data[feature], [75])[0]
                IQR = stats.iqr(data[feature], interpolation = 'midpoint', nan_policy='omit')

                outlier_border = Q3 + 1.5*IQR

                outliers = data[data[feature]>outlier_border]
                if outliers.shape[0] > 0:
                    print(f'o {outliers.shape[0]} datapoints with {feature} > {outlier_border}')
                    if show_details:
                        display(outliers)
                    outliers_info.loc[feature, 'upper_threshold'] = outlier_border
                    outliers_info.loc[feature, 'n_outliers'] = outliers.shape[0]
                else:
                    print(f'o No outliers in {feature}')
                    outliers_info.loc[feature, 'n_outliers'] = 0

            elif distribution == 'left_skewed':
                for col in data.columns:
                    Q1 = np.nanpercentile(data[col], [25])[0]
                    IQR = stats.iqr(data[col], interpolation = 'midpoint', nan_policy='omit')

                    outlier_border = Q1 - 1.5*IQR

                    outliers = data[data[col]>outlier_border]
                    if outliers.shape[0] > 0:
                        print(f'o {outliers.shape[0]} datapoints with {col} > {outlier_border}')
                        if show_details:
                            display(outliers)
                        outliers_info.loc[col, 'lower_threshold'] = outlier_border
                        outliers_info.loc[col, 'n_outliers'] = outliers.shape[0]
                    else:
                        print(f'o No outliers in {col}')
                        outliers_info.loc[col, 'n_outliers'] = 0

            elif (distribution == 'normal') or (distribution == 'heavy_tailed'):
                
                mean = data[feature].mean()
                std = data[feature].std()
                lower_threshold = mean - 3 * std
                upper_threshold = mean + 3 * std
                values_below_3std = data.loc[data[feature] < lower_threshold, feature].values
                values_above_3std = data.loc[data[feature] > upper_threshold, feature].values
                if values_below_3std.shape[0]>0:
                    print(f'o {values_below_3std.shape[0]} datapoints with {feature} < {lower_threshold}')
                if values_above_3std.shape[0]>0:
                    print(f'o {values_above_3std.shape[0]} datapoints with {feature} > {upper_threshold}')
                if values_below_3std.shape[0] == 0 and values_above_3std.shape[0] == 0:
                    print (f'o No outliers in {feature}')

            
                outliers_low = data[data[feature]<lower_threshold]
                outliers_up = data[data[feature]>upper_threshold]
                outliers = pd.concat((outliers_low, outliers_up))

                outliers_info.loc[feature, 'lower_threshold'] = lower_threshold
                outliers_info.loc[feature, 'upper_threshold'] = upper_threshold
                outliers_info.loc[feature, 'n_outliers'] = outliers.shape[0]

                if show_details:
                    display(outliers)
                    
            else: 
                print(f'Impossible to define outliers for {feature} data:  distribution is not in [normal, right-skewed, left-skewed, heavy-tailed]')
                
        else: 
            print(f'Impossible to define outliers for {feature} data: data is not in [continuous, descrete]')

    return outliers_info