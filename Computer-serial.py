import pandas as pd
import numpy as np
import time
import logging as log
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
from scipy.spatial import distance
import random
dir_path = os.path.dirname(os.path.realpath(__file__))

def minMaxScale(vec):
    return (vec-min(vec))/(max(vec)-min(vec))

def minkowskiDistance(v1, v2, p):
    """
    Calculates the Minkowski distance between two vectors
    :param v1: vector 1
    :param v2: vector
    :param p:  The power value for Minkowski distance
    :return:   Sum of distances
    """
    import math
    if p < 1:
        raise ValueError("p must be greater than one for minkowski metric")
    # Getting vector 1 size and initializing summing variable
    size, sum = len(v1), 0

    # Adding the p exponent of the difference of the values of the two vectors
    for i in range(size):
        sum += math.pow(abs(v1[i] - v2[i]), p)
    sum_total_a = math.pow(sum, 1 / p)
    return sum_total_a

def check_cluster(obs, meanMat):
    """
    Finding the minimum distance among a list of distances.
    :param obs:
    :param meanMat:
    :return: Cluster name
    """
    dist = [ minkowskiDistance(obs,mu,3) for mu in meanMat.values]
    min_value = min(dist)
    return dist.index(min_value)+1

def KMeans(k, data):
    """
    Applying knn algorithm on a multivariate data. Get a K number of cluster and fund the optimal centroids
    k: number of clusters
    data: the data to perform the algorithm
    """
    random.seed(2)
    rand_obs = [i for i in random.choices(list(data.index),k = k)]
    meanMat = data.iloc[rand_obs,:]
    meanMat.reset_index(drop=True, inplace=True)
    clusters_list = []
    stop = False
    while(not stop):
        temp_df = data.copy()
        temp_df['check_cluster'] = [check_cluster(x, meanMat) for x in data.values]
        meanMat_star =  temp_df.groupby(['check_cluster']).mean()
        meanMat_star.reset_index(drop=True, inplace=True)
        if meanMat_star.equals( meanMat):
            clusters_list = temp_df['check_cluster']
            stop = True
        else:
            meanMat = meanMat_star

    return clusters_list, meanMat

def fitKmeanForCK(k, data):
    """
    After applying knn algorithm on a multivariate data, claculate the sum squares distances
    :param k: number of clusters
    :param data: the data to perform the algorithm
    :return: List of clusters' name
    """
    print(k)
    data_ = data.copy()
    cluster_list, meanMat = KMeans(k, data_)
    data_['check_cluster'] = cluster_list
    cluster_var = []
    for i in range(0,k):
        temp_ = data_[data_['check_cluster'] == i+1]
        temp_ = temp_.iloc[:,0:data_.shape[1]-1]
        minkowskiDistanceRes = list(temp_.apply(lambda x: minkowskiDistance(x, meanMat.iloc[i,:],p = 2), axis=1))
        cluster_var.append(sum(minkowskiDistanceRes))
    return cluster_var

def scaledClustersVar(list_of_vars, alpha_k = 0.02):
    return [(list_of_vars[k-1] / list_of_vars[0]) + alpha_k * k for k in range(1,len(list_of_vars)+1) ]

def maxAvgVariable(data, by, col):
    """
    For each cluster calculate the average of variable
    :param data:
    :param by:
    :param col:
    :return: The max variable
    """
    temp = data.groupby([by]).mean()
    max_value = max(temp.loc[:,col])
    temp = temp[temp.isin([max_value])].reset_index()
    return temp[temp[col].notna()]

def findOptK(sum_square_for_kCluster, show = False ):
    pass


def optimalK2(slopes):
    """
    This function calculates optimal k in the Elbow graph using KneeLocator
    :param slopes:
    :return:
    """
    np_slopes = np.array(slopes)
    x = np.arange(0, len(slopes))

    np_slopes2 = np_slopes.astype(float)
    x2 = x.astype(float)
    kn = KneeLocator(x2, np_slopes2, curve='convex', direction='decreasing')
    return (kn.knee + 1)

def plotDimensionsPar(cluster_df, k, col_lst):
    """
    Plot 2 dimension by cluster k
    :param cluster_df:
    :param k:
    :param col_lst:
    :return:
    """
    plt.figure()
    patch = []
    for i in range(1, k+1):
        patch.append(mpatches.Patch(color=col_lst[i-1],  label=str(i) ))
        plt.scatter(cluster_df.loc[cluster_df['cluster'] == i+1, 'price'], cluster_df.loc[cluster_df['cluster'] == i+1, 'hd'], color= col_lst[i-1])
    plt.legend(handles=patch)

def heatMap(centroids):
    sns.heatmap(centroids, cmap='Blues_r')

def run():
    log.basicConfig(level=log.DEBUG)
    log.getLogger('matplotlib.font_manager').disabled = True
    pil_logger = log.getLogger('PIL')
    pil_logger.setLevel(log.INFO)

    log.info("This program will run the function without parallelization methods, and will calculate the runtime\n")
    start_time = time.time()

    log.info("File reading\n")
    filename = '\computers_test.csv'
    dataset = pd.read_csv(dir_path + filename)
    dataset = dataset.iloc[:, 1::]
    df   = dataset.copy()
    k = 10

    # Preprocess
    df['laptop'] = df['laptop'].apply(lambda x: 1 if x == 'yes' else 0)
    df['cd']     = df['cd'].apply(lambda x: 1 if x == 'yes' else 0)
    df           = df.apply(minMaxScale)

    # K means clusters
    log.info("apply Kmeans Algorithm\n")
    k_list = [i+1 for i in range(0,k)]
    k_clustering_res = map(lambda x: fitKmeanForCK(x, df), k_list)
    sum_square_for_kCluster = []
    for i in k_clustering_res:
        sum_square_for_kCluster.append(list(i))
    sum_square_for_k = [np.sum(var_list) for var_list in sum_square_for_kCluster]

    log.info("Find the optimal K according the SSE\n")
    opt_k = int(optimalK2(sum_square_for_k))
    plt.plot(np.arange(1, 11), sum_square_for_k)
    plt.axvline(opt_k, color='r')
    log.info(f"The optimal k according to the elbow graph - {opt_k}\n")

    log.info(f"Creating model with {opt_k} cluster\n")
    cluster_df = df.copy()
    optimum_cluster = KMeans(opt_k, cluster_df)
    cluster_df['cluster'] = optimum_cluster[0]

    # 2 first dimension
    log.info("Plots the first 2 dimensions\n")
    col_lst = ['red', 'yellow', 'green', 'blue', 'orange', 'purple', 'black', 'brown', 'olive']
    plotDimensionsPar(cluster_df, opt_k, col_lst)

    # cluster with the highest average price
    log.info("Find the cluster with best average price\n")
    maxAvgVariable(data = cluster_df.loc[:, ['cluster','price']], by = 'cluster', col = 'price')

    # plot a heatmap
    log.info("plot a heatmap\n")
    heatMap(optimum_cluster[1])

    log.info("--- Total run time of serial programming is %s seconds ---" % (time.time() - start_time))
    plt.show()

if __name__ == '__main__':
    run()


