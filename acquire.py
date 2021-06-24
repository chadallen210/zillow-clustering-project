##### IMPORTS #####
import pandas as pd
import numpy as np

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
    print(cols_missing(df))
    