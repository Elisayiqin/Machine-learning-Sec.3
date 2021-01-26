def Q_02(self):
    #Task 2: Standardize daily records of the 3 datasets (confirmed, deaths, recovered)
    # and return all the three scaled dataframes.
    confirmed_scaled = self.Q_01()[0]
    death_scaled = self.Q_01()[1]
    recovered_scaled = self.Q_01()[2]

    ## YOUR CODE HERE
    columns_num = len(confirmed_scaled.columns)
    for i in range(4, columns_num):
        confirmed_scaled.iloc[:, i] = (confirmed_scaled.iloc[:, i] - confirmed_scaled.iloc[:, i].min()) / (confirmed_scaled.iloc[:, i].max() - confirmed_scaled.iloc[:, i].min())
        death_scaled.iloc[:, i] = (death_scaled.iloc[:, i] - death_scaled.iloc[:, i].min()) / (death_scaled.iloc[:, i].max() - death_scaled.iloc[:, i].min())
        recovered_scaled.iloc[:, i] = (recovered_scaled.iloc[:, i] - recovered_scaled.iloc[:, i].min()) / (recovered_scaled.iloc[:, i].max() - recovered_scaled.iloc[:, i].min())
    confirmed_scaled = confirmed_scaled.fillna(0)
    death_scaled = death_scaled.fillna(0)
    recovered_scaled = recovered_scaled.fillna(0)

    return (confirmed_scaled, death_scaled, recovered_scaled)

