
#%%

# Import all required packages

import pandas as pd
import numpy as np
import seaborn as sns

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

# %%

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

df_train.shape, df_test.shape
# %%

df_train.head(10)
#%%

# Plot the mean temp against each year
sns.scatterplot(data = df_train, x = "year", y = "mean_temp_recal")
# The year 1945 has missing values hence the temp is -9999

#%%
# Check the length of rows for 1945
len(df_train.loc[df_train["year"] == 1945])

#%%


# Check the count of days for all years
df_train.groupby("year")["day_in_month"].count().sort_values(ascending=True)

#%%

# Check data for 1945
df_train.loc[df_train["year"] == 1945]["mean_temp_recal"].value_counts()

# %%
# The TG for 1945 is missing (hence -9999)
df_train.loc[df_train["TG"] == -9999]["year"].unique()

#%%

# Remove all values with -9999
indices_to_be_removed = df_train.loc[df_train["TG"] == -9999].index
df_train = df_train[~df_train.index.isin(indices_to_be_removed)]
df_train.shape

#%%
# Remove the year 1945 (because it doesn't have half data available)
df_train = df_train[~df_train.year.isin([1945])]
df_train

#%%

sns.scatterplot(data = df_train, x = "year", y = "mean_temp_recal")

#%%

# Group by the mean temp over a year
df_train_mean_temp = df_train.groupby('year')['mean_temp_recal'].mean().reset_index()
df_train_mean_temp.head(10)

# %%
# df_train_mean_temp[df_train_mean_temp["mean_temp_recal"] == min(df_train_mean_temp.mean_temp_recal)]


# %%

# Plot the mean values per year (Scatterplot)
sns.scatterplot(data= df_train_mean_temp, x = "year", y = "mean_temp_recal")

# %%
# Plot time series with all the years (Lineplot)

sns.lineplot(data = df_train, x = "year", y = "mean_temp_recal")

# %%

# Plot time series with all the months (catplot)
sns.catplot(data=df_train, x="month", y = "mean_temp_recal", kind="box")


# %%

# Reassign the missing values in 1945 with the values from the years before and after

# Remove all values with -9999
indices_to_be_removed = df_train.loc[df_train["TG"] == -9999].index
df_train_missing = df_train[df_train.index.isin(indices_to_be_removed)]
df_train_missing.head(10)

months_days_to_include = df_train_missing.day_in_month.values
days_to_include = df_train_missing.day_in_month.values
# for row in df_train_missing.iterrows():
#     month = row[5]
#     day = row[6]
#     for 

#%%

df_plus_minus_1945 = df_train.loc[(df_train.year >= 1943) & (df_train.year <= 1947) & (df_train.year != 1945)] 

#%%
df_plus_minus_1945_day_month =  df_plus_minus_1945.loc[(df_plus_minus_1945.day_in_month.isin(days_to_include)) & (df_plus_minus_1945["month"].isin(months_to_include))]

df_plus_minus_1945_day_month
# %%

df_train_missing.month.unique()

# %%
len(df_plus_minus_1945_day_month.day_in_month.unique()
)

# %%

len(df_train_missing.day_in_month.unique())
# %%
# %%
df_plus_minus_1945_day_month.year.unique()

# %%

df_train_missing.year.unique()
# %%


df_final = df_plus_minus_1945_day_month.groupby(['month', 'day_in_month'])["mean_temp_recal"].mean().reset_index()
df_final["month_day"] = (df_final["month"]).astype(str) + "_" + (df_final["day_in_month"]).astype(str)

df_final["year"] = 1945

df_final
# %%

# Convert to a dictionary
df_plus_minus_1945_day_month_dict = df_final.to_dict('records')
df_plus_minus_1945_day_month_dict
#%%
# Function to match the boundaries to the stat data
def search(month_day, df_plus_minus_1945_day_month_dict):
    for element in df_plus_minus_1945_day_month_dict:
        if element['month_day'] == month_day:
            output =  element["mean_temp_recal"]
            print(output)
        else:
            print(output)
            continue
    return output

# %%

df_train["month_day"] = (df_train["month"]).astype(str) + "_" + (df_train["day_in_month"]).astype(str)

#%%
df_train.merge(df_final, left_on=['month_day', 'year'], right_on=['month_day', 'year'])


#df_train.head(10)
#%%

df_train.loc[(df_train.year == 1945), "mean_temp_recal"] = df_train.loc[(df_train.year == 1945)]["month_day"].apply(search, df_plus_minus_1945_day_month_dict = df_plus_minus_1945_day_month_dict)

# %%
df_train.loc[(df_train.year == 1945)]

#["mean_temp_recal"].value_counts()# %%

#%%


df_train.loc[(df_train.year == 1945)]["mean_temp_recal"].value_counts()
# %%

# Plot the mean values per year (Scatterplot)
sns.scatterplot(data= df_train, x = "year", y = "mean_temp_recal")

# %%
df_train.loc[(df_train.year == 1945)]["month_day"].apply(search, df_plus_minus_1945_day_month_dict = df_plus_minus_1945_day_month_dict)
# %%

df_mean = df_train[df_train.year.between(1942, 1948, inclusive=False)].loc[df_train["year"] != 1945].loc[(df_plus_minus_1945.day_in_month.isin(days_to_include)) & (df_plus_minus_1945["month"].isin(months_to_include))]
df_means = df_mean.groupby(['month_day'])["mean_temp_recal"].mean().reset_index() 

#%%
df_plus_minus_1945_day_month =  df_plus_minus_1945.loc[(df_plus_minus_1945.day_in_month.isin(days_to_include)) & (df_plus_minus_1945["month"].isin(months_to_include))]
