##### IMPORTS #####
import pandas as pd
import numpy as np
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    percent_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_missing})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

def summarize(df):
    '''
    summarize 
    '''
    print('==============================================')
    print('DataFrame head: ')
    print(df.head(3))
    print('==============================================')
    print('DataFrame info: ')
    print(df.info())
    print('==============================================')
    print('DataFrame description: ')
    print(df.describe())
    num_col = [col for col in df.columns if df[col].dtype!='O']
    cat_col = [col for col in df.columns if col not in num_col]
    print('==============================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_col:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('==============================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('==============================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    
def miss_dup_values(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows, 
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns
    
def remove_outliers(df, col_list, k=1.5):
    for col in col_list:
        
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
    
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
    
        df = df[df[col] > lower_bound]
        df = df[df[col] < upper_bound]
    
    return df

def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.75):
    '''
    takes in a df and amount of required proportion for each row and column
    returns df with after dropping rows and columns that do not meet required proportions
    '''
    # dealing with columns
    col_threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh=col_threshold, inplace=True)
    
    # dealing with rows
    row_threshold = int(round(prop_required_row * len(df.columns),0))
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    
    return df

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy variables of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df = df.drop(columns = ['regionidcounty'])
    return df

def create_features(df):
    '''
    create features for analysis:
    age, tax rate, acres and bath-to-bed ratio
    '''
    # create 'age' column (2017 - 'yearbuilt')
    df['age'] = 2017 - df.yearbuilt
    
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    
    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    return df

def prepare_zillow(df):
    
    # set index to parcelid
    df.set_index('parcelid', drop=True, inplace=True)
    
    # filter by propertylandusetypeid
    df = df[df.propertylandusetypeid == 261]
    
    # drop columns and rows with missing values
    df = handle_missing_values(df)
    
    # create feature for analysis
    df = wrangle_zillow.create_features(df)
    
    # drop columns with redundant information and unneeded columns
    df = df.drop(columns=['propertylandusetypeid', 'heatingorsystemtypeid', 'id', 'buildingqualitytypeid', \
                          'calculatedbathnbr',  'finishedsquarefeet12', 'fullbathcnt', 'lotsizesquarefeet', \
                          'propertycountylandusecode', 'propertyzoningdesc', 'censustractandblock', \
                         'rawcensustractandblock', 'roomcnt', 'unitcnt', 'assessmentyear', \
                          'propertylandusedesc','transactiondate', 'heatingorsystemdesc', 'regionidcity', \
                          'regionidzip', 'yearbuilt'])
    
    # rename columns
    df = df.rename(columns={'bathroomcnt': 'bathrooms', 'bedroomcnt': 'bedrooms', 'calculatedfinishedsquarefeet': 'square_feet', \
                            'fips': 'county_code', 'structuretaxvaluedollarcnt': 'building_value', 'taxvaluedollarcnt': 'appraised_value', \
                            'landtaxvaluedollarcnt': 'land_value', 'taxamount': 'taxes'})
    
    # remove outliers
    df = df[((df.bathrooms > 0) & (df.bathrooms <= 7) & (df.bedrooms > 0) & (df.bedrooms <= 7) & 
               (df.square_feet < 10000) & (df.acres < 20) & (df.taxrate < 10))]
    
    # drop any left over nulls
    df = df.dropna()
 
    return df

def split_zillow(df, target):
    '''
    this function takes in the zillow dataframe
    splits into train, validate and test subsets
    then splits for X (features) and y (target)
    '''
    
    # split df into 20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    
    # split train_validate into 30% validate, 70% train
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    
    # Split with X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
