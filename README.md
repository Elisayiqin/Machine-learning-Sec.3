# Machine-learning-Sec.3
第三节-新冠肺炎的数据分析

## Background

The worldwide highly contagious outbreak of severe respiratory illness, fever caused by the novel COVID-19 virus (also known as the Corona virus) is disrupting us in many extents of our daily lives. However dangerous the impact of it into the humanity, we need to be strong, stay healthy, and solve this assignment dedicated to understanding the spread of the virus little better.

On a daily basis, Johns Hopkins University is collecting number of confirmed cases, deaths and how many have recovered so far, and they put together all the information in a visually appealing map as shown in Figure 1. However, they put the dataset behind all this visualization at a publicly accessible github repository located at : https://github.com/CSSEGISandData/COVID-19.git

## Dataset Description

In the “dataset/” you will find 3 csv files: 

* time_series_19-covid-Confirmed.csv 
  * Contains information about confirmed cases in provinces/states of a particular country.
* time_series_19-covid-Deaths.csv
  * Contains information about cases of deaths in provinces/states of a particular country.
* time_series_19-covid-Recovered.csv
  * Contains information about cases recovered in provinces/states of a particular country.

Each of the 3 CSV files contain the following information:

* Province/State: China - province name; US/Canada/Australia/ - city name, state/province name; Others - name of the event (e.g., "Diamond Princess" cruise ship); other countries - blank.
* Country/Region: country/region name conforming to.
* Lat : Latitude of the province/state. You can discard this variable.
* Long: Longitude of the province/state. You can discard this variable.
* Cumulative aggregates after the daily records of the corresponding cases (Confirmed, Deaths, Recovered). 

## Project Skeleton Description

Q_01.py	Task 1（数据处理-分析不同国家的每日新增人数）:

For each of the 3 dataframes update the dataframes from aggregated daily cases, replacing it with non-aggregate daily recordings, meaning for a particular day put how many new cases were recorded instead of putting total number of cases so far until that day. Return the 3 updated dataframes: [confirmed, deaths, recovered]	10

Q_02.py	Task 2（标准化数据库）: 

Standardize daily records of the 3 datasets (confirmed, deaths, recovered) and return all the three scaled dataframes.	10

Q_03.py	Task 3（根据Kmeans算法原理自行设计kmeans不使用第三方库-用Euclidean距离计算两点之间的距离）:

Implement kmean(X, k, max_iter,random_state) function that takes 3 parameters:
* X, a dataframe (similar to any one of the dataframes you worked with in Q_01-Q_02)
*	k, an integer denoting number of clusters the method is going to compute.
*	max_iter, another integer denoting a convergence criterion you should set in your kmeans() so that number of centroid updates does not exceed that number, max_iter.
*	random_state, an integer number that is used in seeding the random number generator at the beginning of the function definition. Please do not remove that line. 

Use Euclidean distance metric to compute distances between data points. Please do not use kmeans() or similar library functions from sklearn or any available packages in Python. Returns a tuple of a dataframe and a list: (cluster_centroids, class_labels), where the dataframe format of each is described below:

Cluster_centroids:

Label, 1/22/20,…, 3/10/20

C1, _, …, _

C2, _, …, _

Here, ith row will contain the cluster label (defined in the cluster_centroids) for sample i (according to the ordering of the given dataframe), that denotes the membership of the ith sample to any of the k clusters.

Q_04.py	Task 4（设计SSE score 算法）: 

Implement sse( X, cluster_centroids, cluster_labels) function that calculates SSE score for the clustering result. 
*	X, a dataframe (similar to any one of the dataframes you worked with in Q_01-Q_02)
*	cluster_centroids, dataframe as in Q_04
*	cluster_labels, list as in Q_04

Returns the Intra-cluster distance in terms of the Sum of Squared Error (SSE).

Q_05.py	Task 5（利用elbow分析kmeans算法里最好的参数值k）: 

Given 3 dataframes : confirmed_scaled, deaths_scaled, recovered_scaled, computed in Q_02,  find the best k values for doing kmeans on the three datasets. Use the Elbow technique. Please use kmeans() defined by you in Q_03 and sse() defined by you in Q_04.

Returns 3 dataframes: confirmed_results, deaths_results, recovered_results, each follows the following format (example only, please do not rely on the values I put in the format):

k,SSE,isBest

2,_,No

3,_,No

4,_,Yes

…

Q_06.py	Task 6（设计分级聚类算法）: 

Implement hierarchical_clustering() function that takes 2 parameters:
*	X, a dataframe (similar to any one of the dataframes you worked with in Q_01-Q_02)
*	distance_metric, a string that can any of the  3 distance metrics in to consideration: {“cosine”, “euclidean”, “cityblock”,”max-norm”} while doing the hierarchical “agglomerative” clustering. Please note: cosine is a similarity measure, and the other 3 are distance measures.

Please do not use library functions that does the hierarchical clustering from sklearn or any available packages in Python.

It returns a dendrogram as a (N-1) x 3 dimensional numpy array containing N-1 arrays, each containing 3 numbers, where N is the total number of samples in X.
(i)	ID/index of cluster 1, 
(ii)	ID/index of cluster 2, 
(iii)	merging height where cluster 1 and cluster 2 were merged according to the agglomerative clustering algorithm. 

In case of singleton clusters, the ID/Index of the cluster will be the row number (0-based indexing) of the sample in the dataset, X, otherwise, create a new cluster id in the format of p:q, where p is the id of cluster 1 and q is the id of cluster2.


Q_07.py	Task 7（设计cophenetic distance算法）: 

Implement the cophenetic_distance() function. The function is going to using the dataset X and the dendrogram_data (similar to something returned from Q_06), and calculate the cophenetic distance. Return: cophenetic distance

Q_08.py	Task 8（设计剪枝算法来得到需要的几类）:

Input: the dataset X (similar to the scaled dataframe), the dendrogram_data (similar to something returned from Q_06), and k, integer denoting number of clusters we need. So, you need to take a cut at a certain height to obtain k number of clusters. 

Return:. Returns a tuple of a dataframe, a list, and SSE, cut_height: (cluster_centroids, class_labels, SSE, cut_height), where the dataframe format of each is described below:

Here, ith row will contain the cluster label (defined in the cluster_centroids) for sample i (according to the ordering of the given dataframe), that denotes the membership of the ith sample to any of the k clusters. and SSE is the intra-cluster distance of the clustering result, and cut_height is the value of the height where you cut the dendrogram to obtain the k clusters.

