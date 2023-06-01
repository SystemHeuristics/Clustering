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

#Question 2
#Cluster people having heart disease
# Load data
df = pd.read_csv('Preprocessed_Data.csv',sep=',')
df = df.drop(df.columns[[0]],axis = 1)
print(df.head())
print(df.keys())

Elbow_M = KElbowVisualizer(KMeans(), k=(2,15))
Elbow_M.fit(df)
Elbow_M.show()

for k in range(2,15):
    model = KMeans(k)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(df) 
    #visualizer.show()  
    
#it gives good cluster at k=9
#plotting clusters for k=9
kmeans = KMeans(n_clusters= 10)
#predict the labels of clusters.
label = kmeans.fit_predict(df)
 
print(label)
df["Cluster"]=label
print(df)
df.to_csv("dfk10.csv")


#changing attributes


#Initiating PCA to reduce dimentions features to 3
pca = PCA(n_components=3)
pca.fit(df)
PCA_df = pd.DataFrame(pca.transform(df), columns=(["feature1","feature2","feature3"]))

# A 3D Projection Of Data In The Reduced Dimension
plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"])
plt.title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()

Elbow_M = KElbowVisualizer(KMeans(), k=(5,13))
Elbow_M.fit(PCA_df)
Elbow_M.show()

kmeans = KMeans(n_clusters= 8)
#predict the labels of clusters.
label_pca = kmeans.fit_predict(PCA_df)
 
print(label_pca)
PCA_df["Cluster"]=label_pca
print(PCA_df)
PCA_df.to_csv("PCA_df.csv")
for k in range(5,13):
    model = KMeans(k)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(PCA_df)
    #visualizer.show()
    
    
#question 3 part D
st=time.time()
kmeans = KMeans(init="random", n_clusters=7, n_init=10, max_iter=100, random_state=42)
kmeans.fit(PCA_df)
time_taken = time.time()-st
print("TIme taken to perform Kmeans: ",time_taken,"miliseconds")
print("SSE:",kmeans.inertia_) # minimum SSE: 145.21824910533712
# Final locations of the centroid
print("Kmeans Optimized Clusters:",kmeans.cluster_centers_)
# The number of iterations required to converge
print("Number of iterations for convergence:",kmeans.n_iter_)  #k=5 iterations

#saving SSE for each iteration
#cond = KMeans(init="random", n_clusters=8, n_init=10, max_iter=100, random_state=42)
sse = []
for k in range(3, 13):
    kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=100, random_state=42)
    kmeans.fit(PCA_df)
    sse.append(kmeans.inertia_)
print("SSE is:",sse) #minimum SSE=34.3500469022975

#plotting SSE
plt.style.use("fivethirtyeight")
plt.plot(range(3, 13), sse)
plt.xticks(range(3, 13))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

pl = sns.scatterplot(data = PCA_df,x=PCA_df["feature1"], y=PCA_df["feature2"],hue=label_pca)
pl.set_title("Cluster's Profile for KMeans Based On Feature1 And Feature2")
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,8))
plt.axes(projection='3d').scatter(PCA_df["feature1"], PCA_df["feature2"], PCA_df["feature3"], c=label_pca)
plt.title("3D Plot Of The Clusters by PCA:")
plt.show()