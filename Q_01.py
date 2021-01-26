def Q_01(self):
    #Task 1: For each of the 3 dataframes update the dataframes from aggregated daily cases,
    # replacing it with non-aggregate daily recordings, meaning for a particular day put how
    # many new cases were recorded instead of putting total number of cases so far until that day.
    # And, the function returns a tuple of the 3 updated dataframes: (confirmed, deaths, recovered]

    confirmed_noagg = self.confirmed
    deaths_noagg = self.deaths
    recovered_noagg = self.recovered
    ## YOUR CODE HERE ##

    columns_num = len(confirmed_noagg.columns)

    for i in range(columns_num-1, 4, -1):

        confirmed_noagg.iloc[:, i] = confirmed_noagg.iloc[:,i] - confirmed_noagg.iloc[:,i-1]
        deaths_noagg.iloc[:, i] = deaths_noagg.iloc[:, i] - deaths_noagg.iloc[:, i - 1]
        recovered_noagg.iloc[:, i] = recovered_noagg.iloc[:, i] - recovered_noagg.iloc[:, i - 1]

    # confirmed_noagg.to_csv('confirmed_noagg.csv',index=False)
    # deaths_noagg.to_csv('deaths_noagg.csv', index=False)
    # recovered_noagg.to_csv('recovered_noagg.csv', index=False)

    return (confirmed_noagg, deaths_noagg, recovered_noagg)