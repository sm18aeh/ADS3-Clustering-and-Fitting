# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:43:53 2023

@author: sm18aeh
"""

import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt

def read_file(file_name):
    """
    Takes a filename as a parameter
    reads from said file and creates two dataframes:
    -df having empty data removed and unnecessary colums removed
    -countries_df being a transposed version of df
    returns the two dataframes, in order:
    -countries_df
    -df
    """
    #converted WB file from xls to xlsx in order to avoid
    #xlrd module conflict
    df = pd.read_excel(file_name,
                       sheet_name="Data",header=3)
    
    #Removing empty years (no data for all countries at x year column)
    df.dropna(how="all", axis=1, inplace=True)
    #dropping unnecessary columns
    df = df.drop(columns = ["Country Code","Indicator Name","Indicator Code"])
    #Transposing the dataframe with countries as colums
    countries_df = df.set_index("Country Name").T
    
    return countries_df, df


def k_clustering(df_x,df_y,year,labels):
    """
    Takes two dataframes df_x and df_y
    Uses the year parameter to extract the data from
    the dataframes at specified year
    """
    year = str(year)
    #obtaining the data for the selected year for each dataframe
    df_x = df_x[["Country Name", year]].rename(columns={year:labels[0]})
    df_y = df_y[["Country Name", year]].rename(columns={year:labels[0]})
    #merging into a single dataframe
    xy_df = pd.merge(df_x,df_y, on="Country Name")
    #dropping all countries with empty values
    xy_df = xy_df.dropna(how="any")
    #plotting the unclustered scatter graph
    plt.figure()
    plt.scatter(xy_df.iloc[:,1],xy_df.iloc[:,2])
    plt.title(year)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()
    
    #clustering using kmeans
    c_count = 4
    kmeans = cluster.KMeans(c_count)
    kmeans.fit(xy_df.drop(["Country Name"], axis=1))
    k_labels = kmeans.labels_
    xy_df["labels"] = k_labels
    filename = labels[0] + "vs" + labels[1] + ".csv"
    xy_df.to_csv(filename)  #saving to file
    centres = kmeans.cluster_centers_
    print("Silhouette score:",
          skmet.silhouette_score(xy_df.drop(["Country Name"], axis=1),
                                 k_labels))
    grouped_df = xy_df.groupby("labels")
    
    plt.figure()
    #plotting by cluster
    for label,group in grouped_df:
        plt.scatter(group.iloc[:,1],
                    group.iloc[:,2],
                    label=label)
    #plotting cluster centers
    for i in range(c_count):
        xc, yc = centres[i, :]
        plt.plot(xc, yc, "dk", markersize=8, alpha=0.6)
    plt.title(year)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    pop_df, pop_val_df = read_file("Population growth (%).xlsx")
    gdp_df, gdp_val_df = read_file("WB GDP Per Capita.xlsx")
    co2_df, co2_val_df = read_file("WB CO2 Emissions KT Per Capita.xlsx")
    
    k_clustering(pop_val_df,gdp_val_df,2010,
                 ["Population Growth (%)","GDP Per Capita (USD)"])
    k_clustering(pop_val_df,gdp_val_df,1980,
                 ["Population Growth (%)","GDP Per Capita (USD)"])

        
        
        
        
        