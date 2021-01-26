def Q_09(self, X, clustering_labels):
    # Task 9: Please follow requirements as described in the document.

    ### YOUR CODE HERE ###
    import pandas as pd
    import numpy as np
    den = hierarchical_clustering(X, distance_metric='euclidean')
    cophi = cophenetic_distance(X, den)
    coph = cophi.to_numpy()
    for i in range(len(coph[0,:])):
        names = X.iloc[i, 0]
        if names == 'Colorado':
            m = np.argmin(coph[:,i])
            A = X.iloc[m, 0]
        elif names == 'Washington':
            n = np.argmin(coph[:,i])
            B = X.iloc[n, 0]
        elif names == 'New York':
            p = np.argmin(coph[:,i])
            C = X.iloc[p, 0]
        elif names == 'California':
            q = np.argmin(coph[:,i])
            D = X.iloc[q, 0]


    Y = self.Q_02()[1]
    den2 = hierarchical_clustering(Y, distance_metric='euclidean')
    coph2 = cophenetic_distance(Y, den2)
    coph3 = coph2.to_numpy()
    for j in range(len(coph3[0,:])):
        name2 = X.iloc[j, 0]
        if name2 == 'Washington':
            g = np.argmin(coph3[:,j])
            E = X.iloc[g, 0]




    return [A, B, C, D, E]


def cophenetic_distance(X, dendrogram_data):
    #  Using the dataset X and the dendrogram_data
    #  (similar to something returned from Q_06, calculate the cophenetic distance.
    import pandas as pd
    import numpy as np
    coph_dist = pd.DataFrame(None, columns = np.arange(len(X.index)), index = np.arange(len(X.index)))

    ## YOUR CODE HERE

    for i in range(len(dendrogram_data)):
        for j in range(len(dendrogram_data[i][0])):
            row = dendrogram_data[i][0][j]
            for k in range(len(dendrogram_data[i][1])):
                col = dendrogram_data[i][1][k]
                coph_dist.iloc[row,col] = dendrogram_data[i][2]

    return coph_dist.T

def hierarchical_clustering(X, distance_metric = "euclidean"):
    # Please follow requirements as described in the document.
    dendrogram_data = []

    ## YOUR CODE HERE
    import numpy as np
    import pandas as pd

    def cosinesimilar(vector1, vector2):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def euclDistance(vector1, vector2):
        vector1 = vector1.to_numpy()
        vector2 = vector2.to_numpy()
        return np.sqrt(sum(np.power(vector2 - vector1, 2)))

    def city_block_dis(vector1, vector2):
        d1 = np.sum(np.abs(x - y))
        return d1

    def maxnorm_dis(vector1, vector2):
        d2 = np.max(np.abs(x - y))
        return d2

    if distance_metric == 'euclidean':
        X = X.iloc[: , 4:]
        (r, l) = X.shape
        cluster = [[i] for i in range(0, r)]

        # calculate the distance matrix
        d_df = pd.DataFrame()
        for i in range(0, r):
            vector1 = X.loc[i, :]
            d_df.loc[i, i] = None
            for j in range(i + 1, r):
                vector2 = X.loc[j, :]
                d = euclDistance(vector1, vector2)
                d_df.loc[i, j] = d
                d_df.loc[j, i] = d
        print(type(d_df))
        while len(cluster) > 1:
            mincol = d_df.stack().idxmin()
            dmin = d_df.iloc[mincol[0], mincol[1]]
            left = cluster[mincol[0]][:]
            right = cluster[mincol[1]][:]
            dendrogram_data.append([left, right, dmin])
            for i in cluster[mincol[1]]:
                cluster[mincol[0]].append(i)
            del cluster[mincol[1]]
            d_df.iloc[mincol[0],:] = d_df.iloc[[mincol[0], mincol[1]], :].min(axis = 0)
            d_df.iloc[mincol[0], mincol[0]] = None
            d_df.iloc[:, mincol[0]] = d_df.iloc[mincol[0],:].T
            d_df = d_df.drop(index = mincol[1], columns= mincol[1])
            d_df = d_df.reset_index(drop=True)
            d_df = d_df.T.reset_index(drop=True).T

    if distance_metric == 'cityblock':
        X = X.iloc[: , 4:]
        (r, l) = X.shape
        cluster = [[i] for i in range(0, r)]

        # calculate the distance matrix
        d_df = pd.DataFrame()
        for i in range(0, r):
            vector1 = X.loc[i, :]
            d_df.loc[i, i] = None
            for j in range(i + 1, r):
                vector2 = X.loc[j, :]
                d = city_block_dis(vector1, vector2)
                d_df.loc[i, j] = d
                d_df.loc[j, i] = d
        print(type(d_df))
        while len(cluster) > 1:
            mincol = d_df.stack().idxmin()
            dmin = d_df.iloc[mincol[0], mincol[1]]
            left = cluster[mincol[0]][:]
            right = cluster[mincol[1]][:]
            dendrogram_data.append([left, right, dmin])
            for i in cluster[mincol[1]]:
                cluster[mincol[0]].append(i)
            del cluster[mincol[1]]
            d_df.iloc[mincol[0],:] = d_df.iloc[[mincol[0], mincol[1]], :].min(axis = 0)
            d_df.iloc[mincol[0], mincol[0]] = None
            d_df.iloc[:, mincol[0]] = d_df.iloc[mincol[0],:].T
            d_df = d_df.drop(index = mincol[1], columns= mincol[1])
            d_df = d_df.reset_index(drop=True)
            d_df = d_df.T.reset_index(drop=True).T

    if distance_metric == 'max-norm':
        X = X.iloc[: , 4:]
        (r, l) = X.shape
        cluster = [[i] for i in range(0, r)]

        # calculate the distance matrix
        d_df = pd.DataFrame()
        for i in range(0, r):
            vector1 = X.loc[i, :]
            d_df.loc[i, i] = None
            for j in range(i + 1, r):
                vector2 = X.loc[j, :]
                d = maxnorm_dis(vector1, vector2)
                d_df.loc[i, j] = d
                d_df.loc[j, i] = d
        print(type(d_df))
        while len(cluster) > 1:
            mincol = d_df.stack().idxmin()
            dmin = d_df.iloc[mincol[0], mincol[1]]
            left = cluster[mincol[0]][:]
            right = cluster[mincol[1]][:]
            dendrogram_data.append([left, right, dmin])
            for i in cluster[mincol[1]]:
                cluster[mincol[0]].append(i)
            del cluster[mincol[1]]
            d_df.iloc[mincol[0],:] = d_df.iloc[[mincol[0], mincol[1]], :].min(axis = 0)
            d_df.iloc[mincol[0], mincol[0]] = None
            d_df.iloc[:, mincol[0]] = d_df.iloc[mincol[0],:].T
            d_df = d_df.drop(index = mincol[1], columns= mincol[1])
            d_df = d_df.reset_index(drop=True)
            d_df = d_df.T.reset_index(drop=True).T

    if distance_metric == 'cosine':
        X = X.iloc[: , 4:]
        (r, l) = X.shape
        cluster = [[i] for i in range(0, r)]

        # calculate the distance matrix
        d_df = pd.DataFrame()
        for i in range(0, r):
            vector1 = X.loc[i, :]
            d_df.loc[i, i] = None
            for j in range(i + 1, r):
                vector2 = X.loc[j, :]
                d = cosinesimilar(vector1, vector2)
                d_df.loc[i, j] = d
                d_df.loc[j, i] = d
        print(type(d_df))
        while len(cluster) > 1:
            mincol = d_df.stack().idxmax()
            dmin = d_df.iloc[mincol[0], mincol[1]]
            left = cluster[mincol[0]][:]
            right = cluster[mincol[1]][:]
            dendrogram_data.append([left, right, dmin])
            for i in cluster[mincol[1]]:
                cluster[mincol[0]].append(i)
            del cluster[mincol[1]]
            d_df.iloc[mincol[0],:] = d_df.iloc[[mincol[0], mincol[1]], :].max(axis = 0)
            d_df.iloc[mincol[0], mincol[0]] = None
            d_df.iloc[:, mincol[0]] = d_df.iloc[mincol[0],:].T
            d_df = d_df.drop(index = mincol[1], columns= mincol[1])
            d_df = d_df.reset_index(drop=True)
            d_df = d_df.T.reset_index(drop=True).T

    return dendrogram_data