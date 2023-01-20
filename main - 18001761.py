# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:43:53 2023

@author: sm18aeh
"""

import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import scipy.optimize as opt #curve_fit
import numpy as np
import itertools as iter

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


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper  

def exp_growth(t, scale, growth, t0):
    """
    Computes exponential function with scale
    and growth as free parameters
    """
    f = scale * np.exp(growth * (t- t0))
    return f

def logistics(t, scale, growth, t0):
    """ 
    Computes logistics function with scale, growth rate
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def cos(t, scale, growth, t0):
    """
    Computes cos function with scale, growth rate
    and time of peak as parameters
    """
    
    f = scale * np.cos(growth * (t - t0))
    return f


def k_clustering(df_x,df_y,year,c_count,labels):
    """
    Takes two dataframes df_x and df_y
    Uses the year parameter to extract the data from
    the dataframes at specified year
    groups into c_count clusters using kmeans
    takes labels to label the axes
    Output:
        One unclustered scatter plot
        One clustered scatter plot with centroids
        Prints the silhouette score of the clustering
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
    
def fitting(df, function, p0_val, forecast_range, y_label):
    """
    Takes the dataframe to perform the fitting
    Takes the fitting function to be used
    Takes the estimated covariance parameters (p0_val)
    Takes the number of years to forecast
    Takes the y_label string to label the y axis
    Performs function fitting based on the passed parameters
    Outputs:
        One plot with the passed data only
        One plot with data, fitting function, forecast and error ranges 
    """
    #converting the years from string to int
    df.index = df.index.astype(int)
    plt.figure()
    plt.plot(df,"x")
    plt.ylabel(y_label)
    plt.xlabel("Year")
    plt.title(df.columns[0])
    plt.show()
    
    popt, covar = opt.curve_fit(function,
                                df.index.values,
                                df.iloc[:,0],
                                p0=p0_val)
    #adds a new column containing the fitted function
    df["fit"] = function(df.index, *popt)
    #calculating sigma
    sigma = np.sqrt(np.diag(covar))
    #getting the error ranges
    low, high = err_ranges(df.index.values, function, popt, sigma)
    #creating an integer list containing the years from the
    #last in the data
    forecast_years = [i for i in range(df.index.max(),
                                       (df.index.max() + forecast_range))]
    #generates the forecasted values
    forecast_vals = function(forecast_years, *popt)
    #error ranges for the forecasted values
    f_low, f_high = err_ranges(forecast_years, function, popt, sigma)
    #plotting data, fitting function, forecast and error ranges
    plt.figure()
    plt.plot(df.iloc[:,0], "x", label="Data")
    plt.plot(df.index, df["fit"], label="Fit")
    plt.plot(forecast_years,forecast_vals,label="Forecast")
    plt.fill_between(df.index, low, high, alpha=0.5)
    plt.fill_between(forecast_years, f_low, f_high, alpha=0.2,color="g")
    plt.ylabel(y_label)
    #plt.ticklabel_format(useOffset=False)
    plt.xlabel("Year")
    plt.title(df.columns[0])
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    #reading data
    pop_df, pop_val_df = read_file("Population growth (%).xlsx")
    gdp_df, gdp_val_df = read_file("WB GDP Per Capita.xlsx")
    co2_df, co2_val_df = read_file("WB CO2 Emissions KT Per Capita.xlsx")
    
    #performing kmeans clustering
    k_clustering(pop_val_df,gdp_val_df,2010,4,
                 ["Population Growth (%)","GDP Per Capita (USD)"])
    k_clustering(pop_val_df,gdp_val_df,1980,4,
                 ["Population Growth (%)","GDP Per Capita (USD)"])
    
    #creating fitting functions and getting forecast for the next 40 years
    fitting(pop_df["Japan"].to_frame(), cos,
                (1.5, 0.11, 1975), 40, "Pop Growth (%)")
    fitting(co2_df["Japan"].to_frame(),cos,
            (9.5,0.05,1995), 40, "CO2 KT per capita")
    fitting(gdp_df["Japan"].to_frame(),logistics,
            (4e5,0.04,1995), 40, "GDP Per Capita (USD)")
    
    fitting(pop_df["India"].to_frame(), cos,
                (1.5, 0.07, 1975), 40, "Pop Growth (%)")
    fitting(gdp_df["India"].to_frame(),exp_growth,
            (500,0.02,2000), 40, "GDP Per Capita (USD)")
        
        
        
        
        
        