def Q_05(self, confirmed_scaled, deaths_scaled, recovered_scaled):
    # Please follow requirements as described in the document.
    import pandas as pd

    confirmed_results = pd.DataFrame(columns=["k", "SSE", "isBest"])
    deaths_results = pd.DataFrame(columns=["k", "SSE", "isBest"])
    recovered_results = pd.DataFrame(columns=["k", "SSE", "isBest"])

    #### YOUR CODE HERE####
    ## For confirmed_result
    DatascaleX = confirmed_scaled
    dif = 0
    times = 1.5
    k = 1
    ## To get the position of elbow, set the times threshold = 2
    while dif < times:
        cluster_centroids1, cluster_labels1 = kmeans(DatascaleX, k)
        sse_score_1 = sse(DatascaleX, cluster_centroids1, cluster_labels1)
        cluster_centroids2, cluster_labels2 = kmeans(DatascaleX, k + 1)
        sse_score_2 = sse(DatascaleX, cluster_centroids2, cluster_labels2)
        cluster_centroids3, cluster_labels3 = kmeans(DatascaleX, k + 2)
        sse_score_3 = sse(DatascaleX, cluster_centroids3, cluster_labels3)
        dif_1 = sse_score_1 - sse_score_2
        dif_2 = sse_score_2 - sse_score_3
        dif  = dif_1 / dif_2
        if dif > times:
            confirmed_results.loc[k - 1, :] = [k, sse_score_1, "No"]
            confirmed_results.loc[k, :] = [k, sse_score_1, "Yes"]
            break
        confirmed_results.loc[k - 1, :] = [k, sse_score_1, "No"]
        confirmed_results.loc[k, :] = [k, sse_score_1, "No"]
        k = k + 1

    ## For deaths_scaled
    DatascaleY = deaths_scaled
    dif = 0
    times = 10
    k = 1
    ## To get the position of elbow, set the times threshold = 1.5
    while dif < times:
        cluster_centroids4, cluster_labels4 = kmeans(DatascaleY, k)
        sse_score_1 = sse(DatascaleY, cluster_centroids4, cluster_labels4)
        cluster_centroids5, cluster_labels5 = kmeans(DatascaleY, k + 1)
        sse_score_2 = sse(DatascaleY, cluster_centroids5, cluster_labels5)
        cluster_centroids6, cluster_labels6 = kmeans(DatascaleY, k + 2)
        sse_score_3 = sse(DatascaleY, cluster_centroids6, cluster_labels6)
        dif_1 = sse_score_1 - sse_score_2
        dif_2 = sse_score_2 - sse_score_3
        dif = dif_1 / dif_2
        if dif > times:
            deaths_results.loc[k - 1, :] = [k, sse_score_1, "No"]
            deaths_results.loc[k, :] = [k, sse_score_1, "Yes"]
            break
        deaths_results.loc[k - 1, :] = [k, sse_score_1, "No"]
        deaths_results.loc[k, :] = [k, sse_score_1, "No"]
        k = k + 1

    ## For recovered_results
    DatascaleZ = recovered_scaled
    dif = 0
    times = 2
    k = 1
    ## To get the position of elbow, set the times threshold = 2
    while dif < times:
        cluster_centroids7, cluster_labels7 = kmeans(DatascaleZ, k)
        sse_score_1 = sse(DatascaleZ, cluster_centroids7, cluster_labels7)
        cluster_centroids8, cluster_labels8 = kmeans(DatascaleZ, k + 1)
        sse_score_2 = sse(DatascaleZ, cluster_centroids8, cluster_labels8)
        cluster_centroids9, cluster_labels9 = kmeans(DatascaleZ, k + 2)
        sse_score_3 = sse(DatascaleZ, cluster_centroids9, cluster_labels9)
        dif_1 = sse_score_1 - sse_score_2
        dif_2 = sse_score_2 - sse_score_3
        dif = dif_1 / dif_2
        if dif > times:
            recovered_results.loc[k - 1, :] = [k, sse_score_1, "No"]
            recovered_results.loc[k, :] = [k, sse_score_1, "Yes"]
            break
        recovered_results.loc[k - 1, :] = [k, sse_score_1, "No"]
        recovered_results.loc[k, :] = [k, sse_score_1, "No"]
        k = k + 1


    return (confirmed_results, deaths_results, recovered_results)



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



def sse(X, cluster_centroids, cluster_labels):
    # Please follow requirements as described in the document.

    sse_score = 0

    ## YOUR CODE HERE
    import numpy as np
    X = X.iloc[: , 4:]
    distance = np.zeros(X.shape[0])

    for k in range(len(cluster_centroids.index)):
        cluster_samples = []
        center = cluster_centroids.iloc[k, 1:]
        for i in range(len(cluster_labels)):
            if cluster_labels.iloc[i,0] == cluster_centroids.iloc[k,0]:
                cluster_samples.append(i)
        for j in cluster_samples:
            sample = X.loc[j, :]
            error = sample - center
            distance[j] = np.linalg.norm(error)
        # distance[cluster_samples] = np.linalg.norm(X.iloc[cluster_samples, :] - cluster_centroids.iloc[k, 1:], axis=1)

    sse_score = np.sum(np.square(distance))



    return sse_score
