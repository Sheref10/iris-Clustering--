import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas_profiling as pp
import matplotlib
matplotlib.use('tkagg')
#Load the dataset
iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(iris_data.head())

#Data exploration
print(iris_data.info)
print(iris_data.describe())
report=pp.ProfileReport(iris_data)
report.to_file('output.html')
#Data Cleaning
print(iris_data.duplicated().sum())
print(iris_data.drop_duplicates(inplace=True))
print(iris_data.duplicated().sum())
print(iris_data.isnull().sum())
iris_data.corr().style.background_gradient(cmap='green')
iris_data.boxplot()
plt.show()
# Finding the optimum number of clusters for k-means classification
x=iris_data.iloc[:,:].values
#print(x)

wcss=[]
wcss_describe=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    str = "wcss {} = {}"
    wcss_describe.append(str.format(i,kmeans.inertia_))
print(wcss)
print(wcss_describe)


#Plotting the WCSS of N_clusters for the optimal n_clusters
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('number of clustering')
plt.ylabel('cluster sum of squares')
plt.show()

#based on the elbow method the optimal number for clustring is 3 , this is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.

#Modeling
kmeans=KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#Visualize the clusters of Model
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()

