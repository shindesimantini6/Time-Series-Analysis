#%%

# Import all required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# %%

# Read the text file as csv file
temp_berlin = pd.read_csv("./ECA_blended_custom_Berlin_Mitte_mean_temp/TG_STAID004563.txt", skiprows=19)


# %%

# Check the head of the file
temp_berlin.head(10)
# temp_berlin.columns

#%%

# Rename columns to something more nice
temp_berlin = temp_berlin.rename(columns={" SOUID": "SOUID", '    DATE':'DATE', 
'   TG':'TG', ' Q_TG':'Q_TG'})

# Convert DATE to datetime format
temp_berlin['DATE'] = pd.to_datetime(temp_berlin['DATE'], format='%Y%m%d')

# Add a year column
temp_berlin["year"] = temp_berlin['DATE'].dt.year

# Add a month column
temp_berlin["month"] = temp_berlin['DATE'].dt.month

# Add a day column
temp_berlin["day_in_month"] = temp_berlin['DATE'].dt.day

temp_berlin.head(10)

# %%

# Convert the temparature to real temperatures in 
temp_berlin["mean_temp_recal"] = temp_berlin["TG"] * 0.1

# %%
temp_berlin.head(10)

#%%

# Split the train and test data

df_train = temp_berlin[:-365]

df_test = temp_berlin[-365:]

df_train["month_day"] = (df_train["month"]).astype(str) + "_" + (df_train["day_in_month"]).astype(str)

# %%

# Remove all month days with -9999
months_days_to_include = df_train.loc[df_train["TG"] == -9999]["month_day"].values


# %%

df_plus_minus_1945_day_month =  df_train.loc[(df_train["year"].between(1942, 1948, inclusive=False))  & (df_train["year"] != 1945) 
& (df_train.month_day.isin(months_days_to_include))]


# %%
# Calculate the means of the temps for all months and days 

df_means = df_plus_minus_1945_day_month.groupby(['month_day'])["mean_temp_recal"].mean().reset_index() 

# %%
month_day = df_means["month_day"].values

mean_temp_recal = df_means["mean_temp_recal"].values

dict_missing = dict(zip(month_day, mean_temp_recal))
for ein in dict_missing:
    print(ein)

#%%

# Convert it to a dict
df_means_dict = df_means.to_dict()
df_means_dict
# %%

# Create a function to match the mean to the corresponding months

def search(month_day):
    print(month_day)
    return dict_missing.get(month_day)
        
# %%
df_train.loc[(df_train.year == 1945) & (df_train.month_day.isin(months_days_to_include)), "mean_temp_recal"] = df_train.loc[(df_train.year == 1945) & (df_train.month_day.isin(months_days_to_include))]["month_day"].apply(search)


# %%
df_train.loc[(df_train.year == 1945) & (df_train.month_day.isin(months_days_to_include))]
df_train["yr_month"] = df_train["DATE"].astype(str).str[0:7]


#%% 

# Calculate the trend of the time series
def fit_linear_regression(X, y):
    m = LinearRegression()
    return m.fit(X, y)   

def get_seasonal_dummies(df, dummy_column):
    # Add a column with the time step 
    df["timestep"] = list(range(len(df)))
    # Add dummies seasonal values
    seasonal_dummies = pd.get_dummies(df[dummy_column],prefix=dummy_column)
    seasonal_columns = seasonal_dummies.columns.values
    df = df.join(seasonal_dummies)
    return df, seasonal_columns

#%%
def get_seasonal_trend(df, target_value):
    seasonal_trend_columns = ['month_1', 'month_2', 'month_3', 'month_4' ,'month_5' ,'month_6' ,'month_7',
 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', "timestep"]
    X = df[seasonal_trend_columns]
    y = df[target_value]
    m = fit_linear_regression(X, y)
    predicted_values = m.predict(X)
    return predicted_values

def get_remainder_plot_pcaf(df, target_value, seasonal_trend):
    plot_pacf(df['remainder'])
    plt.xlabel('lags')
    return df["remainder"]


#%%

df_train, seasonal_dummies = get_seasonal_dummies(df_train, 'month')

# %%

df_train["seasonal_trend"] = get_seasonal_trend(df_train, 'mean_temp_recal')

df_train['remainder'] = df_train["mean_temp_recal"] - df_train["seasonal_trend"]
# %%
# Plot the predicted trend and the original
sns.lineplot(data = df_train, x= "DATE", y = "mean_temp_recal")
sns.lineplot(data = df_train, x= "DATE", y = "seasonal_trend")

# %%

get_remainder_plot_pcaf(df_train, 'mean_temp_recal', "seasonal_trend")

# %%

def create_lag_columns(df, n_lags):
    for n in n_lags:
        df[f"lag{n}"] = (df['remainder'].shift(n)).astype(float)
        return 
