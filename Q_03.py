def kmeans(X, k, max_iter=300, random_state=12345):
    # Please follow requirements as described in the document.
    import random
    import pandas as pd
    import numpy as np
    random.seed(random_state)  # please do not call this method anywhere else
    cluster_centroids = []
    cluster_labels = []

    ## YOUR CODE HERE ##
    # calculate Euclidean distance
    def euclDistance(vector1, vector2):
        vector1 = vector1.to_numpy()
        vector2 = vector2.to_numpy()
        return np.sqrt(sum(np.power(vector2 - vector1, 2)))

    # init centroids with random samples

    def initCentroids(dataSet, k):
        numsamples, dim = dataSet.shape
        cluster_centroids = pd.DataFrame(columns = dataSet.columns)
        for i in range(k):
            index = int(random.uniform(0, numsamples))
            cluster_centroids.loc[i, :] = dataSet.loc[index, :]
        return cluster_centroids

    dataSet = X.iloc[:, 4:]
    iter_num = 0
    # k-means cluster

    numsamples = dataSet.shape[0]
    cluster_labels = pd.DataFrame(0, index = np.arange(numsamples), columns = ["Class_labels"])
    clusterChanged = True

    ## step 1: init centroids
    cluster_centroids = initCentroids(dataSet, k)

    while clusterChanged:
        if iter_num >= max_iter:
            break
        clusterChanged = False
        ## for each sample
        for i in range(numsamples):  # range
            minDist = 100000.0
            minIndex = 0

            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(cluster_centroids.loc[j, :], dataSet.loc[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if cluster_labels.iloc[i, 0] != minIndex:
                clusterChanged = True
                cluster_labels.iloc[i, :] = minIndex

        ## step 4: update centroids
        for j in range(k):
            cluster_samples = []
            for row in cluster_labels["Class_labels"]:
                if row == j:
                    cluster_samples.append(cluster_labels.index[row])

            # cluster_samples = list()
            # cluster_samples = cluster_labels["Class_labels"].loc[lambda x: x == j].index
            pointsInCluster = dataSet.iloc[cluster_samples, :]
            cluster_centroids.loc[j, :] = pointsInCluster.mean(axis=0)
        iter_num += 1

    print(iter_num)
    print(clusterChanged)

    # revise the format
    label = pd.DataFrame(columns= ["Label"])
    for v in range(0, len(cluster_centroids.index)):
        label.loc[v,:] = "C" + str(v+1)

    cluster_centroids.insert(0, "Label ", label, allow_duplicates= False)

    cluster_labels = cluster_labels + 1
    for i in range(len(cluster_labels)):
        cluster_labels.iloc[i,0] = "C" + str(cluster_labels.iloc[i,0])

    return (cluster_centroids, cluster_labels)


def Q_03(self):
    # Task 3: Implement kmeans above
    X = self.Q_02()[2]
    k = 4
    cluster_centroids, cluster_labels = kmeans(X, k)
    return (cluster_centroids, cluster_labels)
    pass

