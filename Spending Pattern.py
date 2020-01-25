# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:46:08 2019

@author: Krishanu Joarder
"""

# =============================================================================
# Setting the Envoirnment
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
import warnings as w
w.filterwarnings('ignore')

os.chdir('D:\\Krishanu\\Ivy Videos\\Python- Arpendu\\Assignments\\Clustering')
dataset = pd.read_csv("Mall_Customers.csv")

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
dataset.columns
plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'])
#From the scatter plot itself we can recognize that the Annual Income and spending score can be divided into five cluster

plt2=plt.scatter(dataset['Age'],dataset['Spending Score (1-100)'])
#Number of clusters cannot be stated among age and spending score from scatter plot
# =============================================================================
# Split the data into training and test sets
# =============================================================================
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size = 0.2, random_state = 0)



print(train.head())
train_stat = pd.DataFrame(train.describe()).reset_index()

print(test.head())
test_stat = pd.DataFrame(test.describe()).reset_index()


#Checking missing values
print(train.isna().sum())
print(test.isna().sum())

#Spending count with respect to Gender:
Spending_gender = train[['Genre', "Spending Score (1-100)"]].groupby(['Genre'], as_index=False).mean().sort_values(by="Spending Score (1-100)", ascending=False)

#Spending count with respect to Age:
Spending_age= train[['Age', "Spending Score (1-100)"]].groupby(['Age'], as_index=False).mean().sort_values(by="Spending Score (1-100)", ascending=False)


#Spending count with respect to Annual Income:
Spending_Annualincome= train[['Annual Income (k$)', "Spending Score (1-100)"]].groupby(['Annual Income (k$)'], as_index=False).mean().sort_values(by="Spending Score (1-100)", ascending=False)


# =============================================================================
# Testing the accuracy of Annual Income on Spending Score
# =============================================================================


X_train=train.iloc[:, [3, 4]].values
X_test=test.iloc[:, [3, 4]].values


# Using the elbow method to find the optimal number of clusters.
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans_train = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_train.fit(X_train)
    wcss.append(kmeans_train.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method (Train Dataset)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt3=plt.show()



from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans_test = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_test.fit(X_test)
    wcss.append(kmeans_test.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method (Test Dataset)')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt4=plt.show()

#Using elbow method we find that WCSS drops significantly after five clusters

kmeans_train = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans_train = kmeans_train.fit_predict(X_train)

kmeans_test = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans_test = kmeans_test.fit_predict(X_test)


correct = 0
for i in range(len(X_train)):
    predict_me = np.array(X_train[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans_train.predict(predict_me)
    if prediction[0] == y_kmeans_train[i]:
        correct += 1

print("Accuracy of Kmeans is " + str(correct/len(X_train)))


kmeans_test = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans_test = kmeans_test.fit_predict(X_test)


correct = 0
for i in range(len(X_test)):
    predict_me = np.array(X_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans_test.predict(predict_me)
    if prediction[0] == y_kmeans_test[i]:
        correct += 1

print("Accuracy of Kmeans is " + str(correct/len(X_test)))



# =============================================================================
# Visualising the clusters
# =============================================================================

plt.scatter(X_train[y_kmeans_train == 0, 0], X_train[y_kmeans_train == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[y_kmeans_train == 1, 0], X_train[y_kmeans_train == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[y_kmeans_train== 2, 0], X_train[y_kmeans_train == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_train[y_kmeans_train == 3, 0], X_train[y_kmeans_train == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_train[y_kmeans_train == 4, 0], X_train[y_kmeans_train == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans_train.cluster_centers_[:, 0], kmeans_train.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers (Train Dataset)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt5=plt.show()



plt.scatter(X_test[y_kmeans_test == 0, 0], X_test[y_kmeans_test == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_test[y_kmeans_test == 1, 0], X_test[y_kmeans_test == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_test[y_kmeans_test == 2, 0], X_test[y_kmeans_test == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_test[y_kmeans_test == 3, 0], X_test[y_kmeans_test == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_test[y_kmeans_test == 4, 0], X_test[y_kmeans_test == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans_test.cluster_centers_[:, 0], kmeans_test.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers (Test Dataset)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt6=plt.show()

#In both the clusters of Annual Income and spending score we can observe that none of the values are overlapping with each other. Therefore we can use this model to correctly predict Spending score wrt Annual Income





















