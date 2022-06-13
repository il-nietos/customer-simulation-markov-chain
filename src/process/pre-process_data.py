#----------------------------------------------------*
# Steps
'''
1. import packages
2. read in data (csv files)
3. Data cleaning and intial feature engineering
4. Further data engineering
- add checkout timestamp for each customer
- add entrance timestamp for each customer
- resample the data (to get continuous minute-to-minute data for each customer)
- create a column 'next location' (needed to create transition matrix)

5. Save data

Input: raw data: 5 csv files (Monday.csv, Tuesday.csv, Wednesday.csv, Thursday.csv, Friday.csv)
Output: processed data (ready for analysis): clean_data.csv
'''

#----------------------------------------------------*
# -- 1. Import packages --

import os
import pandas as pd
import numpy as np

import glob

import datetime
from datetime import timedelta
from pandas.api.types import is_datetime64_any_dtype as is_datetime
#----------------------------------------------------*
# -- 2. Read in data --

# Set path var to the current directory
print(os.getcwd())

path = ('data/raw')

# Place all csv files into list all_files
all_files = glob.glob(os.path.join(path, "*.csv"))

# Create var date for parsing dates in following cell
date = ['timestamp']

# Read in each csv file in the list to pandas, specify parsing dates (list comprehension)
each_file = (pd.read_csv(f, sep = ';', parse_dates = date, index_col = 0) for f in all_files)

# Concat all dataframes
df = pd.concat(each_file)

#----------------------------------------------------*

# -- 3. Data-cleaning and initial feature engineering (FE) --

# # Check that index is datatime datatype: validation function that raises type error if index is wrong data type
def validate(date_var):
    if is_datetime(date_var):
        print(f'\u001b[32mGOOD!\u001b[0m the column is {type(date_var)} type') # prints GOOD! in green
    else:
        raise TypeError(f' var must be a datetime64, not a {type(date_var)}')

validate(df.index)

# Sort the dataset according to datetime index (important for time series plotting! )
df = df.sort_index()

# Create timestamp copy column (for convenience)
df['timestamp_copy'] = df.index


# Define function that creates new vars: weekday, day, and weekdayname
def date_cols(df):
    df['weekday'] = df.timestamp_copy.dt.weekday
  #  df['day'] = df.index.day
    df['dayname'] = df.timestamp_copy.dt.day_name()
    return df

# Define function that creates a new column - a unique id per each customer
def unique_id(df):
    df['unique_id'] = df.customer_no.astype(str) + '_' + df.weekday.astype(str)
    return df


# Define function that creates new column for when the customer enters the
def start_end_time(df):
    df['start'] = df.groupby('unique_id')['timestamp_copy'].transform('min')
    df['end'] = df.groupby('unique_id')['timestamp_copy'].transform('max')
    df['duration_total'] = abs(df['end'].dt.minute- df['start'].dt.minute)
    return df


# Call cleaning and FE functions:

df = date_cols(df)
df = unique_id(df)
df = start_end_time(df)

# Sort values
df.sort_values(by=['weekday', 'customer_no'], inplace =True)


#----------------------------------------------------*

# 4. Further feature engineering

# Create a copy of miniversion of dataframe
df_copy = df.head(100).copy()

# Create a full copy of dataframe
df_test=df.copy()


# -- 4.1. Checkout non-checked out customers and create entrance as first location--

df.sort_values(by=['customer_no', 'timestamp_copy'], inplace=True)


# Define function that returns a dictionary with new timestamped rows for each non-checkedout customer
# WHY: this way each customer's last location is checkout

def checkout_customers(df):

    '''
    What this function does: for each customer that did not check-out, artificially add a new row one minute after their last location,
    and mark this location to be checkout
    '''

    last_loc = df.groupby('unique_id')['location'].last() #pandas series with latest loc of each customer
    unique_l = df.groupby('unique_id')['unique_id'].last() # same for if
    last_time = df.groupby('unique_id')['timestamp_copy'].last() # same for timestamp
    dictionary = {'location': [], 'unique_id': [], 'timestamp_copy': []} # create empty dict in which to store values

    for i, loc in enumerate(last_loc): # Loop through values
        un_id= unique_l[i]
        time_l = last_time[i]

        if loc != 'checkout': # if the customer's last loc was not checkout, then:

            #This dictionary will later be 1 new row for each 28 non-checkout customers
            dictionary['location'].append('checkout') # mark location as checkout
            dictionary['unique_id'].append(un_id) # mark customer id as the customer id
            dictionary['timestamp_copy'].append(time_l + timedelta(minutes=1)) # increment latest time by 1 minute with timedelta
    # Return dictionary with 3 keys (location, unique_id, timestamp_copy)
    return dictionary


# 4.2 Create new timestamp for each customer with entrance as location

# Define function that returns a dictionary with new timestamped rows for each customer - first location set to enrance
def entrance_customers(df):

    unique_f = df.groupby('unique_id')['unique_id'].first() # same for if
    first_time = df.groupby('unique_id')['timestamp_copy'].first() # same for timestamp
    dict_entrance = {'location': [], 'unique_id': [], 'timestamp_copy': []} # create empty dict in which to store values

    for i, loc in enumerate(first_time): # Loop through values
        un_id= unique_f[i]
        time_f = first_time[i]
        dict_entrance['location'].append('entrance') # mark location as checkout
        dict_entrance['unique_id'].append(un_id) # mark customer id as the customer id
        dict_entrance['timestamp_copy'].append(time_f - timedelta(minutes=1)) # increment latest time by 1 minute with timedelta
    # Return dictionary with 3 keys (location, unique_id, timestamp_copy)
    return dict_entrance

# Call entrance and checkout functions
dict_checkout = checkout_customers(df)
dict_ent = entrance_customers(df)

# Append checkout dict to dataframe
df = df.append(pd.DataFrame(dict_checkout), ignore_index=True)

# Append entrance dict to dataframe
df = df.append(pd.DataFrame(dict_ent), ignore_index=True)

# -- resort, reset index, fill in missing values after appending dictionaries

df.sort_values(by=['customer_no', 'timestamp_copy'], inplace=True)
df.reset_index(inplace=True)

# Re-run some feature engineering functions to fill out empty values -- only columns
df = date_cols(df)
#df = exit_time(df)
df = start_end_time(df)

# Sort values again
df.sort_values(by=['customer_no', 'timestamp_copy'], inplace=True)

# Back, and forward-fill empty customer_no values by unique_id
df['customer_no'] = df.groupby('unique_id')['customer_no'].ffill().bfill()

# Sort values again
df.sort_values(by=['customer_no', 'timestamp_copy'], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

#----------------------------------------------------*

# -- 4.3. Resample data on minute basis and forward fill empty values --


# Reset df.timestamp_copy as index in order to forward fill
df.set_index(['timestamp_copy'], drop= False, inplace=True)

# Rename index (avoid situation where index and a column in df called the same, could be problemo)
df.index.rename('datetime', inplace=True)

# Resample by minute - for each customer their time in the supermarket is tracked on a
#minute basis from the time they enter to the time they check-out, values are forward-filled

df = df.groupby(df.unique_id.rename('cust_index')).resample('1min').ffill()

df.reset_index(level=0, inplace=True)

# Sort values again
df.sort_values(by=['customer_no', 'timestamp_copy'], inplace=True)

# Update timestamp
df['timestamp_copy'] = df.index

# Drop old index
df.drop(['cust_index'], axis = 1, inplace =True)

#----------------------------------------------------*
# 4.4. Create transition shift for Markov chains


# Create shift - next location: this creates a new column which includes
#the next location the customer goes to (should be NaN in their last location)

df['location_next'] = df.groupby(['unique_id'])['location'].shift(-1)

# Fill NaN values in location with checkout
# Location_next after checkout will be NaN. This results in wrong calculations for the transition time --> fill NaN with checkout
df['location_next'].fillna('checkout', inplace = True)

#----------------------------------------------------*

#5.  Save processed data -- this data is ready for analysis
df.to_csv('data/processed/clean_data.csv')
