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

PCA_df = pd.read_csv('PCA_df.csv',sep=',')
PCA_df = PCA_df.drop(PCA_df.columns[[0,4]],axis = 1)
print(PCA_df.head())
print(PCA_df.keys())



#Initiating the BBSCAN Clustering model 
st=time.time()
DB = DBSCAN(eps=0.1, min_samples=5).fit(PCA_df)
time_taken = time.time()-st
print("TIme taken to perform DBSCAN: ",time_taken,"miliseconds")
# fit model and predict clusters
print("DBSCAN labels: ",DB.labels_)
#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d')
ax.scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"], c=DB.labels_)
ax.set_title("The Plot Of The Clusters by DBSCAN model ")
plt.show()


p = sns.scatterplot(data=PCA_df, x="feature1", y="feature2", hue=DB.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters of DBSCAN')
plt.show()