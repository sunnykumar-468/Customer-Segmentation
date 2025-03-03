# Customer-Segmentation
The goal of this project is to develop a machine learning model that can predict the appropriate customer segment for new potential customers based on their characteristics. 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Uploading of Dataset
df = pd.read_csv("/content/train (1).csv")

# To Modify The DataFrame
df = df.iloc[:,1:]
df.head()

# To Retrieve Information About the Dataset
df.info()

# To Drop Null Values
df = df.dropna()

#  To Randomly Select a Subset of 2,000 rows from the DataFrame
df_sampled = df.sample(n=2000, random_state=42)
df_sampled

from sklearn.preprocessing import LabelEncoder

# Loop through all columns to check if they are categorical (object type)
for column in df_sampled.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_sampled[column] = le.fit_transform(df_sampled[column])

# Display the updated DataFrame
print(df_sampled)

df_sampled_corr = df_sampled.corr()
df_sampled_corr

# Seaborn Library is used for Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30, 24))  # Set the size of the plot
sns.heatmap(df_sampled_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

def winsorize_column(df, column, lower_percentile=1, upper_percentile=90):
    lower_limit = df[column].quantile(lower_percentile / 100)
    upper_limit = df[column].quantile(upper_percentile / 100)
    print(column)
    median_value = df[column].median()

# Replace values below the lower limit with the median
    df[column] = df[column].apply(lambda x: median_value if x < lower_limit else x)
    
# Replace values above the upper limit with the median
    df[column] = df[column].apply(lambda x: median_value if x > upper_limit else x)

    return df
    
for column in df_sampled.columns:
    df_sampled = winsorize_column(df_sampled, column)

df_sampled.plot(kind='box')
df_sampled
