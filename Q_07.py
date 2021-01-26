def cophenetic_distance(self, X, dendrogram_data):
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

def Q_07(self, X):
    # Task 7: Implement the cophenetic_distance() function.
    X = self.Q_02()[0]
    den = self.Q_06(X)
    coph = cophenetic_distance(self, X, den)
    return coph

    # pass