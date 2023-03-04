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
bs_hour = bs.groupby(["hour", "date"])["Start date"].count().reset_index(name = 'total_rentals')

# %%

# sns.lineplot(data=bs_hour, x="date", y = "total_rentals")
# plt.xticks(rotation=90) 

# %%

# Add timestep
bs_hour['timestep'] = list(range(len(bs_hour)))

# convert datetime to datetime format
bs_hour['date'] = pd.to_datetime(bs_hour['date'])
# %%

# Create month column in the df
bs_hour["month"] = bs_hour.date.dt.month

# Add a year column
bs_hour["year"] = bs_hour.date.dt.year

bs_hour["week"] = bs_hour["date"].dt.isocalendar().week

# Add a day column
bs_hour["day"]= bs_hour["date"].dt.day

# Combine the hour, day, year and month columns
bs_hour["date_hour"] = pd.to_datetime(bs_hour[["year", "month", "day", "hour"]])

bs_hour = bs_hour.sort_values(by=["date_hour"])
# %%
# Add dummies for months

seasonal_dummies = pd.get_dummies(bs_hour.week,prefix='week')

df_new = bs_hour.join(seasonal_dummies)

#%%

# Split the data to train and test
df_train = df_new.loc[df_new["year"] != 2013]
#%%
df_test = df_new.loc[df_new["year"] == 2013]

#%%
# Define X and y
X = df_train[['timestep', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7',
       'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13',
       'week_14', 'week_15', 'week_16', 'week_17', 'week_18', 'week_19',
       'week_20', 'week_21', 'week_22', 'week_23', 'week_24', 'week_25',
       'week_26', 'week_27', 'week_28', 'week_29', 'week_30', 'week_31',
       'week_32', 'week_33', 'week_34', 'week_35', 'week_36', 'week_37',
       'week_38', 'week_39', 'week_40', 'week_41', 'week_42', 'week_43',
       'week_44', 'week_45', 'week_46', 'week_47', 'week_48', 'week_49',
       'week_50', 'week_51', 'week_52']]

y = df_train["total_rentals"]

# %%
# Run a Linear Regression
m = LinearRegression()
m.fit(X, y)
# %%
# Use the model to make a prediction
df_train['trend_seasonal'] = m.predict(X)
df_train.head(15)
# %%
# Plot the predicted trend and the original
sns.lineplot(data = df_train, x= "date_hour", y = "total_rentals")
sns.lineplot(data = df_train, x= "date_hour", y = "trend_seasonal")

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

# %%
X2 = df_train[['timestep', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7',
       'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13',
       'week_14', 'week_15', 'week_16', 'week_17', 'week_18', 'week_19',
       'week_20', 'week_21', 'week_22', 'week_23', 'week_24', 'week_25',
       'week_26', 'week_27', 'week_28', 'week_29', 'week_30', 'week_31',
       'week_32', 'week_33', 'week_34', 'week_35', 'week_36', 'week_37',
       'week_38', 'week_39', 'week_40', 'week_41', 'week_42', 'week_43',
       'week_44', 'week_45', 'week_46', 'week_47', 'week_48', 'week_49',
       'week_50', 'week_51', 'week_52', 'lag1']]
y2 = df_train["total_rentals"]    
# %%
# Run a Linear Regression
m3 = LinearRegression()
m3.fit(X2, y2)
# %%

# Create a model with trend and seasonality
df_train['trend_final'] = m3.predict(X2)
df_train.head()

m3.score(X2, y2)
# %%
sns.lineplot(data = df_train, x= "date_hour", y = "total_rentals", label = "rentals_og")
sns.lineplot(data = df_train, x= "date_hour", y = "trend_final", label = "rentals_final")
sns.lineplot(data = df_train, x= "date_hour", y = "remainder", label = "remainder")


# %%

# Group by date i.e. month
bs_hour_date = df_train.groupby("date")["total_rentals", "trend_final"].mean()
# %%
sns.lineplot(data = bs_hour_date, x= "date_hour", y = "total_rentals", label = "rentals_date")
sns.lineplot(data = bs_hour_date, x= "date_hour", y = "trend_final", label = "rentals_final")

# %%

# Work on the test data

X_test = df_test[['timestep', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7',
       'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13',
       'week_14', 'week_15', 'week_16', 'week_17', 'week_18', 'week_19',
       'week_20', 'week_21', 'week_22', 'week_23', 'week_24', 'week_25',
       'week_26', 'week_27', 'week_28', 'week_29', 'week_30', 'week_31',
       'week_32', 'week_33', 'week_34', 'week_35', 'week_36', 'week_37',
       'week_38', 'week_39', 'week_40', 'week_41', 'week_42', 'week_43',
       'week_44', 'week_45', 'week_46', 'week_47', 'week_48', 'week_49',
       'week_50', 'week_51', 'week_52']]
y_test = df_test["total_rentals"]

# %%

# Predict on the test data
df_test["trend_seasonal"] = m.predict(X_test)
# %%

m.score(X_test, y_test)

# %%
df_test["remainder"] = df_test["total_rentals"] - df_test["trend_seasonal"]
df_test["lag1"] = (df_test['remainder'].shift(1)).astype(float)
df_test.loc[(df_test.index == 726), "lag1"] = 109.134449

#%%
sns.lineplot(data = df_test, x= "date_hour", y = "total_rentals", label = "og")
sns.lineplot(data = df_test, x= "date_hour", y = "trend_seasonal", label = "trend_seasonal")
sns.lineplot(data = df_test, x= "date_hour", y = "remainder", label = "remainder")

# %%

X2_test = df_test[['timestep', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7',
       'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13',
       'week_14', 'week_15', 'week_16', 'week_17', 'week_18', 'week_19',
       'week_20', 'week_21', 'week_22', 'week_23', 'week_24', 'week_25',
       'week_26', 'week_27', 'week_28', 'week_29', 'week_30', 'week_31',
       'week_32', 'week_33', 'week_34', 'week_35', 'week_36', 'week_37',
       'week_38', 'week_39', 'week_40', 'week_41', 'week_42', 'week_43',
       'week_44', 'week_45', 'week_46', 'week_47', 'week_48', 'week_49',
       'week_50', 'week_51', 'week_52', 'lag1']]
y2_test = df_test["total_rentals"]    

# %%

# Create a model with trend and seasonality
df_test['trend_final'] = m3.predict(X2_test)
df_test.head()


# %%
m3.score(X2_test, y2_test)


# %%

# Predict the future

bs_combined = df_train.append(df_test)


# %%
# Re-train the model on the whole dataset
X_combined = bs_combined.drop(columns=['total_rentals','week', 'trend_seasonal', 'remainder', 'trend_final', 'hour', 'date', 'month', 'year'])
y_combined = bs_combined['total_rentals']


#%%

#fit a model on the complete (!) dataset => m_combined is then used to predict on X_future (unseen data)
m_combined = LinearRegression()
m_combined.fit(X_combined, y_combined)
# %%
# We want to recreate a future datapoint that corresponds to the next timestep (january 1961); 
# so we will add timestep:
timestep = bs_combined['timestep'].max() + 1
months = [1] 
months.extend([0]*51) # array of zeroes
lag = -205.916400 # lag for the new datapoint/row is the remaidner in the last for of the combined dataset
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
# Prediction for 2014-01-01-00
m_combined.predict(X_future)

# %%


# Check for the stationarity of the remainder

# let's build an AR(1) model for the remainder

ar_1 = AutoReg(X2['remainder'], lags=1, old_names=False).fit()
ar_1.summary(X2)
ar_1.preduct()

#%%

ar_2 = AutoReg(df_train['remainder'], lags=2, old_names=False).fit()
ar_2.summary()

#%%

ar_3 = AutoReg(df_train['remainder'], lags=3, old_names=False).fit()
ar_3.summary()
# %%
