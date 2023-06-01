import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as shc
import time
#from kmodes.kprototypes import KPrototypes

# Load data
df = pd.read_csv('heartdatak.csv',sep=',')
print(df.head())

#check values in each column to know missing values
for column in df:
    print("unique values in\"",column,"\"are:")
    print(df[column].value_counts())

#drop thal, slope, ca as they have majow missing values
#drop class as it is irrelevant for clustering

df.drop(columns=['class','ca', 'thal', 'slope'],inplace=True)


#replace ? with NaN
df.fbs = pd.to_numeric(df.fbs, errors='coerce')
df.oldpeak = pd.to_numeric(df.oldpeak, errors='coerce')
df.exang = pd.to_numeric(df.exang, errors='coerce')
df.thalach = pd.to_numeric(df.thalach, errors='coerce')
df.chol = pd.to_numeric(df.chol, errors='coerce')
df.trestbps = pd.to_numeric(df.trestbps, errors='coerce')
df.restecg = pd.to_numeric(df.restecg, errors='coerce')

#fill missing values with mean
df.fbs.fillna(df.fbs.mean(), inplace=True)
df.oldpeak.fillna(df.oldpeak.mean(), inplace=True)
df.exang.fillna(df.exang.mean(), inplace=True)
df.thalach.fillna(df.thalach.mean(), inplace=True)
df.chol.fillna(df.chol.mean(), inplace=True)
df.trestbps.fillna(df.trestbps.mean(), inplace=True)
df.restecg.fillna(df.restecg.mean(), inplace=True)


#convert datatype to integer
df['fbs'] = df['fbs'].astype('int')
df['trestbps'] = df['trestbps'].astype('int')
df['oldpeak'] = df['oldpeak'].astype('int')
df['exang'] = df['exang'].astype('int')
df['thalach'] = df['thalach'].astype('int')
df['chol'] = df['chol'].astype('int')
df['restecg'] = df['restecg'].astype('int')

for column in df:
    print("unique values in\"",column,"\"are:")
    print(df[column].value_counts())
    
#EDA
df.describe(include='all') 
df.isna().sum()
df.info()

#Correlation
print(df.corr())
plt.matshow(df.corr())
plt.show()

#drawing boxplot to detect outliers
sns.boxplot(x="variable", y="value", data=pd.melt(df))
plt.title('BoxPlot of Data Columns')
plt.show()

fig, axes = plt.subplots(ncols=len(df.columns), figsize=(30,15))
for ax, col in zip(axes, df.columns):
  sns.distplot(df[col], ax=ax)
  plt.tight_layout() 
plt.show()

#convert categorical variabled with one hot encoding
print(df.keys())
df = pd.get_dummies(df, columns=["gender","cp","fbs","restecg","exang"])
print("converted categorical data:\n",df)
print(df.keys())

#scale numerical data
# Using DataFrame.filter() method.
df_continuous = df.filter(['age', 'trestbps', 'chol','thalach', 'oldpeak'], axis=1)

scale_df=MinMaxScaler().fit_transform(df_continuous)
print("scaled data \n:",scale_df)
df.drop(columns=['age', 'trestbps', 'chol','thalach', 'oldpeak'],inplace=True)
print("Dataframe: \n",df)
preprocessed_df = pd.DataFrame(scale_df,columns= df_continuous.columns )
preprocessed_df = pd.concat([preprocessed_df,df],axis=1)
print("Preprocessed Data: \n",preprocessed_df)
preprocessed_df.isna().sum()
# preprocessed_df = preprocessed_df.fillna(0)
print(df.keys())


#drop irrelevant columns
preprocessed_df.drop(columns=[ 'gender_0', 'gender_1'],inplace=True)
print("Data ready for clustering:\n",preprocessed_df)
print(preprocessed_df.keys())
preprocessed_df.to_csv("Preprocessed_Data.csv")
