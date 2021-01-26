def Q_08(self, X, dendrogram_data, k):
    # Task 8: Please follow requirements as described in the document.
    cluster_centroids = []
    class_labels = []
    SSE = -1
    cut_height= -1


    ### YOUR CODE HERE ###
    import pandas as pd
    import numpy as np
    X = X.iloc[:, 4:]
    numsamples = X.shape[0]
    cluster_centroids = pd.DataFrame(columns=X.columns)
    class_labels = pd.DataFrame(0, index = np.arange(numsamples), columns = ["Class_labels"])
    clusters = []
    whole = []

    # convert the dendrogram to cluster sets
    if k == 1:
        whole = [i for i in range(len(X))]
        clusters = [i for i in range(len(X))]
    else:
        for i in range(len(dendrogram_data)-k+1, len(dendrogram_data)):
            if dendrogram_data[i][0] not in whole:
                clusters.append(dendrogram_data[i][0])
                for j in dendrogram_data[i][0]:
                    whole.append(j)
            if dendrogram_data[i][1] not in whole:
                clusters.append(dendrogram_data[i][1])
                for k in dendrogram_data[i][1]:
                    whole.append(k)

    # find out all samples with cluster label
    print(len(clusters))
    for m in range(len(clusters)):
        subcluster =  clusters[m]
        subdata = X.iloc[subcluster, :]
        cluster_centroids.loc[m, :] = subdata.mean(axis=0)
        class_labels.iloc[subcluster, :] = m+1

    label = pd.DataFrame(columns=["Label"])
    for v in range(0, len(cluster_centroids.index)):
        label.loc[v, :] = "C" + str(v + 1)
    cluster_centroids.insert(0, "Label ", label, allow_duplicates=False)

    class_labels = class_labels + 1
    for s in range(len(class_labels)):
        class_labels.iloc[s, 0] = "C" + str(class_labels.iloc[s, 0])

    SSE = sse(X, cluster_centroids, class_labels)
    cut_cluster = len(dendrogram_data) - k + 1
    cut_height = dendrogram_data[cut_cluster][2]


    return (cluster_centroids, class_labels, SSE, cut_height)

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