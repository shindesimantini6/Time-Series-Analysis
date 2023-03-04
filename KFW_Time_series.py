#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg


# %%
# Read csv file

electric_charging_df = pd.read_csv('/home/shinde/Documents/trainings/TU_Data_Science_Chanllenge/Ladesaeulenregister_CSV_3.csv',skiprows=10, sep=';',encoding="ISO-8859-1")

# %%

electric_charging_df["Längengrad"] = (electric_charging_df["Längengrad"]).astype(float)
electric_charging_df["Breitengrad"] = (electric_charging_df["Breitengrad"]).astype(float)


# %%

# Convert DATE to datetime format
electric_charging_df['DATE'] = pd.to_datetime(electric_charging_df['Inbetriebnahmedatum'], format='%d.%m.%Y')

# Add a year column
electric_charging_df["year"] = electric_charging_df.DATE.dt.year

# Add a month column
electric_charging_df["month"] = electric_charging_df['DATE'].dt.month

# electric_charging_df = electric_charging_df.set_index('DATE')

electric_charging_df['timestep'] = list(range(len(electric_charging_df)))

# Drop all columns with null values

electric_charging_df_drop_null_all_col = electric_charging_df.dropna(axis=1)

# %%
electric_charging_bundesland_date = electric_charging_df_drop_null_all_col.groupby(by=["Inbetriebnahmedatum"])["Inbetriebnahmedatum"].count().reset_index(name = 'Number_of_charging_locations')
electric_charging_bundesland_date['Inbetriebnahmedatum'] = pd.to_datetime(electric_charging_bundesland_date['Inbetriebnahmedatum'], format='%d.%m.%Y')

electric_charging_bundesland_date.head(20)
#%%

# Plot the predicted trend and the original
electric_charging_bundesland_date[electric_charging_bundesland_date["Number_of_charging_locations"] > 50].plot(x = "Inbetriebnahmedatum", y = "Number_of_charging_locations")

# %%

# The conclusion is that, the series can't be seasonal because it changes 

electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2010].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2010")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2011].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2011")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2012].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2012")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2013].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2013")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2014].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2014")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2015].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2015")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2016].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2016")
electric_charging_df_drop_null_all_col.loc[electric_charging_df_drop_null_all_col["year"] == 2017].groupby(["month"])["Inbetriebnahmedatum"].count().plot()
plt.legend("2017")


# %%
# Add timestep
electric_charging_bundesland_date['timestep'] = list(range(len(electric_charging_bundesland_date)))

# Create month column in the df
electric_charging_bundesland_date["month"] = electric_charging_bundesland_date.Inbetriebnahmedatum.dt.month

# Add a year column
electric_charging_bundesland_date["year"] = electric_charging_bundesland_date.Inbetriebnahmedatum.dt.year

#%%
# Reduce data for number more than 10

electric_charging_bundesland_date_ten = electric_charging_bundesland_date[electric_charging_bundesland_date["Number_of_charging_locations"] > 50]

# Add dummies for months

seasonal_dummies = pd.get_dummies(electric_charging_bundesland_date_ten.month,prefix='month')

electric_charging_bundesland_date_ten_final = electric_charging_bundesland_date_ten.join(seasonal_dummies)

# %%
# Divide the data into train and test
# Split the train and test data

df_train = electric_charging_bundesland_date_ten_final[electric_charging_bundesland_date_ten_final["year"] != 2022]
df_test = electric_charging_bundesland_date_ten_final[electric_charging_bundesland_date_ten_final["year"] == 2022]

#%%
# Define X and y
X = df_train[['timestep', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']]

y = df_train["Number_of_charging_locations"]
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
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "Number_of_charging_locations")
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "trend_seasonal")

# %%

df_train["remainder"] = df_train["Number_of_charging_locations"] - df_train["trend_seasonal"]

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
X2 = df_train[['timestep', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'lag1']]
y2 = df_train["Number_of_charging_locations"]    
# %%
# Run a Linear Regression
m3 = LinearRegression()
m3.fit(X2, y2)
# %%

# Create a model with trend and seasonality
df_train['trend_final'] = m3.predict(X2)
df_train.head()
# %%
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "Number_of_charging_locations", label = "temp_og")
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "trend_final", label = "temp_final")

#%%

sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "Number_of_charging_locations", label = "og")
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "trend_seasonal", label = "trend_seasonal")
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "remainder", label = "remainder")
sns.lineplot(data = df_train, x= "Inbetriebnahmedatum", y = "trend_final", label = "temp_final")
# %%

X_test = df_test[['timestep', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']]
y_test = df_test["Number_of_charging_locations"]

# Predict on the test data
df_test["trend_seasonal"] = m.predict(X_test)
# %%

m.score(X_test, y_test)

# %%

df_test["remainder"] = df_test["Number_of_charging_locations"] - df_test["trend_seasonal"]
df_test["lag1"] = (df_test['remainder'].shift(1)).astype(float)
df_test.loc[(df_test.index == 12), "lag1"] = 9.934594
#%%
sns.lineplot(data = df_test, x= "Inbetriebnahmedatum", y = "Number_of_charging_locations", label = "og")
sns.lineplot(data = df_test, x= "Inbetriebnahmedatum", y = "trend_seasonal", label = "trend_seasonal")
sns.lineplot(data = df_test, x= "Inbetriebnahmedatum", y = "remainder", label = "remainder")

# %%

X2_test = df_test[['timestep', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'lag1']]
y2_test = df_test["Number_of_charging_locations"]    

# %%

# Create a model with trend and seasonality
df_test['trend_final'] = m3.predict(X2_test)
df_test.head()


# %%
m3.score(X2_test, y2_test)

# %%

# let's build an AR(1) model for the remainder

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

# %%
