#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.linear_model import LinearRegression
# import for plotting
import matplotlib.pyplot as plt
import numpy as np
# statsmodel
import statsmodels
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa import stattools
import datetime
import warnings
warnings.filterwarnings("ignore")

#%%

# With the real data

bs_2011 = pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2011-capitalbikeshare-tripdata.csv")
bs_2012_q1 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2012-capitalbikeshare-tripdata/2012Q1-capitalbikeshare-tripdata.csv")
bs_2012_q2 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2012-capitalbikeshare-tripdata/2012Q2-capitalbikeshare-tripdata.csv")
bs_2012_q3 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2012-capitalbikeshare-tripdata/2012Q3-capitalbikeshare-tripdata.csv")
bs_2012_q4 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2012-capitalbikeshare-tripdata/2012Q4-capitalbikeshare-tripdata.csv")
bs_2013_q1 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2013-capitalbikeshare-tripdata/2013Q1-capitalbikeshare-tripdata.csv")
bs_2013_q2 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2013-capitalbikeshare-tripdata/2013Q2-capitalbikeshare-tripdata.csv")
bs_2013_q3 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2013-capitalbikeshare-tripdata/2013Q3-capitalbikeshare-tripdata.csv")
bs_2013_q4 =  pd.read_csv("/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/Week3/data/2013-capitalbikeshare-tripdata/2013Q4-capitalbikeshare-tripdata.csv")

bs = pd.concat([bs_2011, bs_2012_q1, bs_2012_q2, bs_2012_q3, bs_2012_q4, bs_2013_q1, bs_2013_q2, bs_2013_q3, bs_2013_q4])
# %%

# convert datetime to datetime format
bs['Start date'] = pd.to_datetime(bs['Start date'])

# Create a column for date
bs["date"] = bs['Start date'].dt.date

# Add day as a column
bs["day"] = bs['Start date'].dt.day #date

# Add day as a column
bs["year"] = bs['Start date'].dt.year #date

# Add day as a column
bs["month"] = bs['Start date'].dt.month #date

# Add day as a column
bs["hour"] = bs['Start date'].dt.hour #date

#%%

# Group by for hour and dates
bs_hour = bs.groupby(["hour", "date"])["Start date"].count().reset_index(name = 'total_rentals')

# %%

# convert datetime to datetime format
bs_hour['date'] = pd.to_datetime(bs_hour['date'])

# Create month column in the df
bs_hour["month"] = bs_hour.date.dt.month

# Add a year column
bs_hour["year"] = bs_hour.date.dt.year

# Add a day column
bs_hour["day"]= bs_hour["date"].dt.day

# Combine the hour, day, year and month columns
bs_hour["date_hour"] = pd.to_datetime(bs_hour[["year", "month", "day", "hour"]])

#%%

bs_hour = bs_hour.sort_values(by=["date_hour"], ascending=True)

#%%
sns.lineplot(data=bs_hour, x="date_hour", y = "total_rentals", label = "Total rentals per hour")
plt.xticks(rotation=90) 

# %%

# Add timestep
bs_hour['timestep'] = list(range(len(bs_hour)))

# %%
# Add dummies for months

seasonal_dummies = pd.get_dummies(bs_hour.month,prefix='month')

df_new = bs_hour.join(seasonal_dummies)

#%%

# Split the data to train and test
df_train = df_new.loc[df_new["year"] != 2013]
#%%
df_test = df_new.loc[df_new["year"] == 2013]

#%%
# Define X and y
X = df_train[['timestep', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']]

y = df_train["total_rentals"]

# %%
# Run a Linear Regression
m = LinearRegression()
m.fit(X, y)
# %%
# Use the model to make a prediction
df_train['trend_seasonal'] = m.predict(X)
df_train.head(15)

m.score(X, y)
# %%

# Plot the predicted trend and the original
sns.lineplot(data = df_train, x= "date_hour", y = "total_rentals", label = "Total Rentals")
sns.lineplot(data = df_train, x= "date_hour", y = "trend_seasonal", label = "Fitted Trend Seasonal")
plt.xticks(rotation=90)
plt.legend("Trend Seasonal vs Original Data")

# %%
# Get the remainders for train data
df_train["remainder"] = df_train["total_rentals"] - df_train["trend_seasonal"]

# df_train["lag"]
# %%

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df_train['remainder'])
plt.xlabel('lags')


plot_pacf(df_train['remainder'])
plt.xlabel('lags')

# %%
df_train["lag1"] = (df_train['remainder'].shift(1)).astype(float)
df_train = df_train.dropna()

#%%
sns.lineplot(data = df_train, x= "date_hour", y = "total_rentals", alpha = 0.5, label = "og", color = "red")
sns.lineplot(data = df_train, x= "date_hour", y = "trend_seasonal", alpha = 0.5,label = "trend_seasonal", color = "black")
sns.lineplot(data = df_train, x= "date_hour", y = "remainder", alpha = 0.5, label = "remainder")
plt.xticks(rotation=90) 

# %%
X2 = df_train[['timestep', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'lag1']]
y2 = df_train["total_rentals"]    
# %%
# Run a Linear Regression
m3 = LinearRegression()
m3.fit(X2, y2)
# %%

# Create a model with trend and seasonality and remainder
df_train['trend_final'] = m3.predict(X2)
df_train.head()

m3.score(X2, y2)
# %%
sns.lineplot(data = df_train, x= "date_hour", y = "total_rentals", label = "rentals_og")
sns.lineplot(data = df_train, x= "date_hour", y = "trend_final", label = "rentals_final")
plt.xticks(rotation=90) 
# %%

# Group by date i.e. month
bs_hour_date = df_train.groupby("date")["total_rentals", "trend_final"].mean()
# %%
sns.lineplot(data = bs_hour_date, x= "date", y = "total_rentals", label = "rentals_date")
sns.lineplot(data = bs_hour_date, x= "date", y = "trend_final", label = "rentals_final")
plt.xticks(rotation=90) 
# %%

# Work on the test data

X_test = df_test[['timestep', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']]
y_test = df_test["total_rentals"]

# %%

# Predict on the test data
df_test["trend_seasonal"] = m.predict(X_test)
# %%

m.score(X_test, y_test)

# %%
df_test["remainder"] = df_test["total_rentals"] - df_test["trend_seasonal"]
df_test["lag1"] = (df_test['remainder'].shift(1)).astype(float)
df_test.loc[(df_test.index == 726), "lag1"] = -140.377812

#%%
sns.lineplot(data = df_test, x= "date_hour", y = "total_rentals", label = "Total_rentals")
sns.lineplot(data = df_test, x= "date_hour", y = "trend_seasonal",  label = "trend_seasonal")
#%%
sns.lineplot(data = df_test, x= "date_hour", y = "total_rentals", alpha = 0.5, label = "Total_rentals", color = "red")
sns.lineplot(data = df_test, x= "date_hour", y = "trend_seasonal", alpha = 0.5, label = "trend_seasonal", color = "black")
sns.lineplot(data = df_test, x= "date_hour", y = "remainder", alpha = 0.5, label = "remainder")
plt.xticks(rotation=90) 
# %%

X2_test = df_test[['timestep', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'lag1']]
y2_test = df_test["total_rentals"]    

# %%

# Create a model with trend and seasonality
df_test['trend_final'] = m3.predict(X2_test)
df_test.head()


# %%
m3.score(X2_test, y2_test)

#%%
sns.lineplot(data = df_test, x= "date_hour", y = "total_rentals", label = "Total_rentals")
sns.lineplot(data = df_test, x= "date_hour", y = "trend_final",  label = "trend_seasonal")

#%%
# Group by date i.e. month
bs_test_hour_date = df_test.groupby("date")["total_rentals", "trend_final"].mean()
# %%
sns.lineplot(data = bs_test_hour_date, x= "date", y = "total_rentals", label = "rentals_date")
sns.lineplot(data = bs_test_hour_date, x= "date", y = "trend_final", label = "rentals_final")
plt.xticks(rotation=90) 

# %%

# Predict the future

bs_combined = df_train.append(df_test)


# %%
# Re-train the model on the whole dataset
X_combined = bs_combined.drop(columns=['total_rentals', 'trend_seasonal', 'remainder', 'trend_final', 'hour', 'date', 'month', 'year', 'day', 'date_hour',])
y_combined = bs_combined['total_rentals']


#%%

#fit a model on the complete (!) dataset => m_combined is then used to predict on X_future (unseen data)
m_combined = LinearRegression()
m_combined.fit(X_combined, y_combined)
# %%
# We want to recreate a future datapoint that corresponds to the next timestep (january 1961); 
# so we will add timestep:
timestep = bs_combined['timestep'].max() + 1
months = [1,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # array of zeroes
lag = -221.835662		 # lag for the new datapoint/row is the remaidner in the last for of the combined dataset
# %%
# Create a future data point
X_future = [timestep]
X_future

#%%

X_future.extend(months)
X_future
# %%
X_future.append(lag)
X_future
# %%
X_future = pd.DataFrame([X_future])
X_future.columns = X_combined.columns

X_future
# %%
# Prediction for 1961-01-01
m_combined.predict(X_future)

# %%


# Check for the stationarity of the remainder

# let's build an AR(1) model for the remainder

ar_1 = AutoReg(df_train['remainder'], lags=1, old_names=False).fit()
ar_1.summary()

#%%

ar_2 = AutoReg(df_train['remainder'], lags=2, old_names=False).fit()
ar_2.summary()

#%%

ar_3 = AutoReg(df_train['remainder'], lags=3, old_names=False).fit()
ar_3.summary()
# %%
ar_4 = AutoReg(df_train['remainder'], lags=4, old_names=False).fit()
ar_4.summary()
# %%
