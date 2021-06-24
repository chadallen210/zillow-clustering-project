# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns

import os
from env import host, username, password

##### DB CONNECTION #####
def get_db_url(db, username=username, host=host, password=password):
    
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

##### ACQUIRE ZILLOW #####
def new_zillow_data():
    '''
    gets zillow information from CodeUp db using SQL query
    and creates a dataframe
    '''

    # SQL query
    zillow_query = '''SELECT *
                        FROM properties_2017 prop
                        INNER JOIN (SELECT parcelid, logerror, max(transactiondate) transactiondate
                                    FROM predictions_2017
                                    GROUP BY parcelid, logerror) pred USING (parcelid)
                        LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                        LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                        LEFT JOIN propertylandusetype land USING (propertylandusetypeid)
                        LEFT JOIN storytype story USING (storytypeid)
                        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                        LEFT JOIN unique_properties special USING (parcelid)
                        WHERE prop.latitude IS NOT NULL 
                        AND prop.longitude IS NOT NULL
                        AND transactiondate LIKE '2017%'
                    '''
    
    # reads SQL query into a DataFrame            
    df = pd.read_sql(zillow_query, get_db_url('zillow'))
    
    return df

def get_zillow_data():
    '''
    checks for existing csv file
    loads csv file if present
    if there is no csv file, calls new_zillow_data
    '''
    
    if os.path.isfile('zillow.csv'):
        
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        df = new_zillow_data()
        
        df.to_csv('zillow.csv')
    
    return df

def overview(df):
    '''
    This function returns the shape and info of the df. It also includes a breakdown of the number of unique values
    in each column to determine which are categorical/discrete, and which are numerical/continuous. Finally, it returns
    a breakdown of the statistics on all numerica columns.
    '''
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('----------------------------------')
    print('')
    print(df.info())
    print('----------------------------------')
    print('')
    print('Unique value counts of each column')
    print('')
    print(df.nunique())
    print('----------------------------------')
    print('')
    print('Stats on Numeric Columns')
    print('')
    print(df.describe())