import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import random
import logging as log
import multiprocessing as mp
from functools import partial
import seaborn as sns
from operator import itemgetter
from kneed import KneeLocator
import matplotlib.patches as mpatches
from scipy.spatial import distance
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

def fitKmeanForCK(data, k):
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

def maxAvgVariable(data, k):
    """
    For each cluster calculate the average of variable
    :param data:
    :param by:
    :param col:
    :return: The max variable
    """
    cluster_k_df = data.loc[data['cluster'] == k, 'price']
    return [k, np.mean(cluster_k_df)]

def calcMaxFromList(list_of_max):
    return sorted(list_of_max, key=itemgetter(1), reverse=True)[0]

def returnCunks(chunk):
    return chunk

def read_file(process, chunksize, filename):
    result = pd.DataFrame()
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        for chunk in reader:
            result = pd.concat([result, process.apply(returnCunks, args=(chunk,))], ignore_index=True)
    return result

def plotHeatMap():
    pass

def par_scale(df):
    pool = mp.Pool(processes=df.shape[1])
    col_value_list = [df.loc[:, col] for col in df.columns]
    res = pool.map(minMaxScale, col_value_list)
    temp = pd.concat(res, axis=1, keys=[s.name for s in res])
    pool.close()
    return temp

def parFitKmeans(df, k_list):
    pool = mp.Pool(processes = len(k_list))
    func = partial(fitKmeanForCK, df)
    res = pool.map(func, k_list)
    pool.close()
    return res

def optimalK2(slopes):
    np_slopes = np.array(slopes)
    x = np.arange(0,10)

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
        plt.scatter(cluster_df.loc[cluster_df['cluster'] == i, 'price'], cluster_df.loc[cluster_df['cluster'] == i, 'hd'], color= col_lst[i-1])
    plt.legend(handles=patch)

def avgPricePar(k_list, dataset):
    pool = mp.Pool(processes = len(k_list))
    list_of_priceCluster = pool.map(partial(maxAvgVariable, dataset), [cluster + 1 for cluster in k_list])
    pool.close()
    return calcMaxFromList(list_of_priceCluster)

def heatMap(centroids):
    sns.heatmap(centroids, cmap='Blues_r')


def run(k_to_run, log):
    start_time = time.time()
    PROCESSES = mp.cpu_count() - 10
    log.info('Creating pool with %s processes\n' % PROCESSES)
    filename = dir_path + '\computers_test.csv'
    chunksize = round(5000 / (mp.cpu_count() - 2))

    with mp.Pool(PROCESSES) as pool:
        # read file
        log.info("Reading File to DF\n")
        dataset = read_file(pool, chunksize, filename)
        dataset = dataset.iloc[:, 1::]
        df = dataset.copy()
        k = k_to_run
        df['laptop'] = df['laptop'].apply(lambda x: 1 if x == 'yes' else 0)
        df['cd']     = df['cd'].apply(lambda x: 1 if x == 'yes' else 0)

        # scale
        log.info("Scaling the DF\n")
        for col in df.columns:
            df[col] = pool.apply(minMaxScale, args=(df.loc[:, col],))

        # knn
        log.info("apply Kmeans Algorithm\n")
        k_list = [i + 1 for i in range(0, k)]
        k_clustering_res = [pool.apply(fitKmeanForCK, args=(df, x,)) for x in k_list]
        sum_square_for_k = [np.sum(var_list) for var_list in k_clustering_res]


         # choose best k
        opt_k = int(optimalK2(sum_square_for_k))
        plt.plot(np.arange(1,k+1),sum_square_for_k)
        plt.axvline(opt_k,color='r')
        log.info(f"The optimal k according to the elbow graph{ opt_k}\n")

         # 2 first dimension
        log.info("Creating model with 3 clusterF\n")
        cluster_df = df.copy()
        optimum_cluster = KMeans(k, cluster_df)
        cluster_df['cluster'] = optimum_cluster[0]

        #  Plot the data of the first 2 dimension
        col_lst = ['red', 'yellow', 'green', 'blue', 'orange', 'purple', 'black']
        for i in range(1, opt_k+1):
            plt.scatter(cluster_df.loc[cluster_df['cluster'] == i, 'price'], cluster_df.loc[cluster_df['cluster'] == i, 'hd'], color=col_lst[i-1])

        # cluster with the highest average price
        log.info("Choosing the cluster with the max average price\n")
        list_of_priceCluster = pool.map(partial(maxAvgVariable,cluster_df), [cluster+1 for cluster in range(0,k)])
        max_price_cluster = calcMaxFromList(list_of_priceCluster)
        log.info("The cluster %d has the max average price %s"%(max_price_cluster[0], round(max_price_cluster[1],2)))

        # plot a heatmap
        log.info("\nplot a heatmap\n")
        heatMap(optimum_cluster[1])
        log.info("--- Total run time of multiprocessing' 1 programming is  %s seconds ---" % (time.time() - start_time))
        pool.close()


def run_2(k_to_run, log):
    start_time = time.time()
    PROCESSES = mp.cpu_count() - 10
    log.info('Creating pool with %s processes\n' % PROCESSES)
    filename = dir_path + '\computers_test.csv'
    chunksize = round(5000 / (mp.cpu_count() - 2))

    with mp.Pool(PROCESSES) as pool:
        # read file
        log.info("Reading File to DF")
        dataset = read_file(pool, chunksize, filename)
        dataset = dataset.iloc[:, 1::]
        df = dataset.copy()
        k = k_to_run
        df['laptop'] = df['laptop'].apply(lambda x: 1 if x == 'yes' else 0)
        df['cd'] = df['cd'].apply(lambda x: 1 if x == 'yes' else 0)
        pool.close()

    # scale
    log.info("\nScaling the DF\n")
    df = par_scale(df)

    # knn
    log.info("apply Kmeans Algorithm\n")
    k_list = [i + 1 for i in range(0, k)]
    k_clustering_res = parFitKmeans(df, k_list)
    sum_square_for_k = [np.sum(var_list) for var_list in k_clustering_res]

    # choose best k
    opt_k = int(optimalK2(sum_square_for_k))
    plt.plot(np.arange(1,11),sum_square_for_k)
    plt.axvline(opt_k,color='r')
    log.info(f"The optimal k according to the elbow graph - { opt_k}\n")

    # 2 first dimension
    log.info(f"Creating model with {opt_k} clusterF\n")
    k_list = [i + 1 for i in range(0, opt_k)]
    cluster_df = df.copy()
    optimum_cluster = KMeans(opt_k, cluster_df)
    cluster_df['cluster'] = optimum_cluster[0]

    # Plot the data of the first 2 dimension
    col_lst = ['red', 'yellow', 'green', 'blue', 'orange', 'purple', 'black', 'brown', 'olive']
    plotDimensionsPar(cluster_df, opt_k, col_lst)

    # cluster with the highest average price
    log.info("Choosing the cluster with the max average price\n")
    max_price_cluster = avgPricePar(k_list, cluster_df)
    log.info("The cluster %d has the max average price %s" % (max_price_cluster[0], round(max_price_cluster[1], 2)))

    # # plot a heatmap
    log.info("plot a heatmap\n")
    heatMap(optimum_cluster[1])
    log.info("\n--- Total run time of multiprocessing' 2 programming is  %s seconds ---" % (time.time() - start_time))
    plt.show()


if __name__ == '__main__':
    log.basicConfig(level=log.DEBUG)
    log.getLogger('matplotlib.font_manager').disabled = True
    pil_logger = log.getLogger('PIL')
    pil_logger.setLevel(log.INFO)

    log.info("\nFirst implementation of multithreading")
    # run(10, log)

    log.info("\nSecond implementation of multithreading")
    run_2(10, log)
