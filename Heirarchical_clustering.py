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

#Initiating the Agglomerative Clustering model Complete Linkage
st = time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='complete')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
time_taken = time.time()-st
print("TIme taken to perform complete linkage: ",time_taken,"miliseconds")
#PCA_df["Clusters"] = yhat_AC
print(yhat_AC)

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"], c=yhat_AC, marker='o', cmap = 'viridis')
plt.title("The Plot Of The Clusters by Agglomerative model Complete Linkage")
plt.show()


pl = sns.scatterplot(data = PCA_df,x=PCA_df["feature1"], y=PCA_df["feature2"],hue=yhat_AC)
pl.set_title("Cluster's Profile for Heirarchical Clustering Based On Feature1 And Feature2")
plt.legend()
plt.show()


#dendrogram complete linkage

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms of Complete Linkage")  
dend = shc.dendrogram(shc.linkage(PCA_df, method='complete'))
plt.axhline(y=6, color='r', linestyle='--')

#Initiating the Agglomerative Clustering model Single Linkage
st=time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='single')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
time_taken = time.time()-st
print("TIme taken to perform single linkage: ",time_taken,"miliseconds")
#PCA_df["Clusters"] = yhat_AC
print(yhat_AC)


#Plotting the clusters
fig = plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"], c=yhat_AC)
plt.title("The Plot Of The Clusters by Agglomerative model Single Linkage")
plt.show()

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms of Single Linkage")  
dend = shc.dendrogram(shc.linkage(PCA_df, method='single'))
plt.axhline(y=6, color='r', linestyle='--')

#Initiating the Agglomerative Clustering model Average Linkage
st=time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='average')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
time_taken = time.time()-st
print("TIme taken to perform average linkage: ",time_taken,"miliseconds")
#PCA_df["Clusters"] = yhat_AC
print(yhat_AC)

#Plotting the clusters
fig = plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"], c=yhat_AC, marker='o', cmap = 'viridis')
plt.title("The Plot Of The Clusters by Agglomerative model Average Linkage")
plt.show()

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms of Average Linkage")  
dend = shc.dendrogram(shc.linkage(PCA_df, method='average'))
plt.axhline(y=6, color='r', linestyle='--')
plt.show()
