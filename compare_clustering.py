from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
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
from sklearn.metrics import silhouette_score
PCA_df = pd.read_csv('PCA_df.csv',sep=',')
PCA_df = PCA_df.drop(PCA_df.columns[[0,4]],axis = 1)
print(PCA_df.head())
print(PCA_df.keys())


# Compute the silhouette scores for each algorithm
st=time.time()
kmeans = KMeans(n_clusters=7)
kmeans.fit(PCA_df)
kmeans_silhouette = silhouette_score(PCA_df, kmeans.labels_).round(2)
time_taken = time.time()-st
print("TIme taken to perform Kmeans: ",time_taken,"miliseconds")
print("Silhoutte score for Kmeans: ",kmeans_silhouette)
print("SSE for Kmeans: ",kmeans.inertia_)

#DBSCAN
st=time.time()
dbscan = DBSCAN(eps=0.3)
dbscan.fit(PCA_df)
dbscan_silhouette = silhouette_score(PCA_df, dbscan.labels_).round (2)
time_taken = time.time()-st
print("TIme taken to perform DBSCAN: ",time_taken,"miliseconds")
print("Silhoutte score for DBSCAN: ",dbscan_silhouette)

#Heirarchical Clustering
st=time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='average')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
hc_silhouette = silhouette_score(PCA_df, yhat_AC).round (2)
print("Silhoutte score for Heirarchical clustering average Linkage: ",hc_silhouette)
time_taken = time.time()-st
print("TIme taken to perform average Linkage: ",time_taken,"miliseconds")

#complete linkage
st=time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='complete')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
hc_silhouette = silhouette_score(PCA_df, yhat_AC).round (2)
print("Silhoutte score for Heirarchical clustering Complete Linkage: ",hc_silhouette)
time_taken = time.time()-st
print("TIme taken to perform complete average: ",time_taken,"miliseconds")

#single linkage
st=time.time()
AC = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='single')
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_df)
hc_silhouette = silhouette_score(PCA_df, yhat_AC).round (2)
print("silhoutte score for Heirarchical clustering Single Linkage: ",hc_silhouette)
time_taken = time.time()-st
print("TIme taken to perform single linkage: ",time_taken,"miliseconds")