def sse(self, X, cluster_centroids, cluster_labels):
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

def Q_04(self):
    # Task 4: implement SSE function above.
    confirm_ = self.Q_02()[2]
    centroids = self.Q_03()[0]
    cluster = self.Q_03()[1]
    ssescore = sse(self, confirm_, centroids, cluster)
    return ssescore
    # pass


