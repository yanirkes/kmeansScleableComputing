import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
import threading as thrd
from operator import itemgetter
from scipy.spatial import distance
from kneed import KneeLocator
import logging as log
dir_path = os.path.dirname(os.path.realpath(__file__))

def return_value(data):
    try:
        val = data.value
    except AttributeError:
        print("No value has yet assigned")
    else:
        return val

def worker(data, function, **kwargs):
    function(data, **kwargs)

class MyLocal(thrd.local):
    def __init__(self, value):
        self.value = value

    def append_to_val(self, data):
        self.value.append(data)

    def getData(self):
        return self.value

class optimizeThreadsTime():

    @classmethod
    def testTime(cls,function,  **kwargs):
        start_time = time.time()
        lc_data = function(**kwargs)
        return [time.time() - start_time, lc_data, kwargs]

class testFunction():

    def __init__(self, name, fun, **kwargs):
        self.name = name
        self.fun = fun
        self.kwargs = kwargs
        self.max_parameters = None
        self.best_time = None
        self.test_results = []

    def getMaxParameters(self):
        self.max_parameters = sorted(self.test_results, key=itemgetter(0))[0]

    def getBestTime(self):
        rest_lst = sorted(self.test_results, key=itemgetter(0))[0]
        self.best_time = rest_lst[0]

    def OptMyFunParameters(self, num_of_iter):
        for num_threads in range(1, num_of_iter+1):
            self.kwargs["num_threads"] = num_threads
            self.test_results.append(optimizeThreadsTime.testTime(self.fun, **self.kwargs))


def minkowskiDistance(v1, v2, p, result_list = None):
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
    try:
        # Getting vector 1 size and initializing summing variable
        size, sum = len(v1), 0

        # Adding the p exponent of the difference of the values of the two vectors

        for i in range(size):
            sum += math.pow(abs(v1[i] - v2[i]), p)
        sum_total_a = math.pow(sum, 1 / p)
        if result_list is not None:
            result_list.append(sum_total_a)
    except:
        print(thrd.current_thread(),"\n", "v1: ", v1,"\nv2: ", v2)
        return

    return sum_total_a

def check_cluster( data =None, **kwargs):
    """
    Finding the minimum distance among a list of distances.
    :param obs:
    :param meanMat:
    :return: Cluster name
    """
    obs = kwargs["obs"]
    meanMat = kwargs["meanMat"]
    result_list = [minkowskiDistance(obs, mu, 3) for mu in meanMat.values]
    min_value = min(result_list)
    return result_list.index(min_value) + 1

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
    final_clusters_list = []
    stop = False
    while(not stop):
        temp_df = data.copy()
        temp_df['check_cluster'] = [check_cluster(None,**{ "obs": x, "meanMat": meanMat}) for x in data.values]
        meanMat_star = temp_df.groupby(['check_cluster']).mean()
        meanMat_star.reset_index(drop=True, inplace=True)
        if meanMat_star.equals(meanMat):
            final_clusters_list = temp_df['check_cluster']
            stop = True
        else:
            meanMat = meanMat_star
    print(thrd.current_thread())
    return final_clusters_list, meanMat

def calcMinkowskiForDf(data, **kwargs):
    """
    Calculate for each obs in data the minkowski distance from a given mean matrix.
    :param data:  k, dataset and meanMat
    :param kwargs:
    :return:
    """

    i = kwargs["k"]
    df = kwargs["dataset"]
    meanMat = kwargs["meanMat"]

    temp_ = df[df['check_cluster'] == i + 1]
    temp_ = temp_.iloc[:, 0:df.shape[1] - 1]
    res = sum(list(temp_.apply(lambda x: minkowskiDistance(x, meanMat.iloc[i, :], p=2), axis=1)))
    data.append_to_val(res)
    return

def fitKmeanForCK(data, **kwargs):
    """
    Applying knn algorithm on a multivariate data, and calculates the sum squares distances
    :param k: number of clusters
    :param dataset: the data to perform the algorithm
    :return: List of clusters' name
    """
    data_ = kwargs["dataset"].copy()
    k     = kwargs["k"]
    print(k)

    cluster_list, meanMat = KMeans(k, data_)
    data_['check_cluster'] = cluster_list
    cluster_var = []

    result = []
    th = []
    minkowskiDistanceRes = MyLocal(result)
    for i in range(0,k):
        t = thrd.Thread(target=worker, args=(minkowskiDistanceRes, calcMinkowskiForDf,), kwargs={'k': i, 'dataset': data_, "meanMat": meanMat})
        th.append(t)
    for t_ in th:
        t_.start()
    for i in th:
        i.join()

    data.append_to_val(minkowskiDistanceRes.getData())
    return cluster_var

def maxAvgVariable(data, **kwargs):
    """
    For each cluster calculate the average of variable
    :param data:
    :param by:
    :param col:
    :return: The max variable
    """
    temp_df = kwargs['dataset']
    cluster_k_df = temp_df.loc[temp_df['cluster'] == kwargs['k'], 'price']
    data.append_to_val([ kwargs['k'], np.mean(cluster_k_df)])
    return [ kwargs['k'], np.mean(cluster_k_df)]

def calcMaxFromList(list_of_max):
    return sorted(list_of_max, key=itemgetter(1), reverse=True)[0]

def collectResult(result):
    global results
    results.append(result)

def read_chunk(chunk, data):
    data.append(chunk)

def read_filePar( **kwargs):
    """
    Create parallel env to\and read a csv file by chunks
    :param kwargs: filename and number of threads
    :return:
    """
    filename = kwargs["filename"]
    num_threads = kwargs["num_threads"]
    data = list()
    job = []
    chunksize = round(5000 / (num_threads))
    with pd.read_csv(filename, chunksize=chunksize) as reader:
        for ind, chunk in enumerate(reader):
                thread = thrd.Thread(name='th%s' % ind, target=read_chunk(chunk,data,))
                job.append(thread)
    for j in job:
        j.start()

    for j in job:
        j.join()
    return data

def minMaxScale(vec,col):
    col.append((vec-min(vec))/(max(vec)-min(vec)))

def scaleDataPar(**kwargs):
    """
    Create parallel env to scale the data
    :param kwargs: data
    :return:
    """
    data = kwargs["data"]
    col = list()
    job = []
    for ind, colName in enumerate(data.columns):
            thread = thrd.Thread(name= str(ind), target=minMaxScale(data.loc[:,colName], col))
            job.append(thread)
    for j in job:
        j.start()

    for ind, col_ in enumerate(col):
        data.iloc[:,ind] = col_

    for j in job:
        j.join()
    return data

def applyKmeansAlgorithmPar(**kwargs):
    """
    Create parallel env to apply the kmeans algorithm.
    :param kwargs: k_list and df
    :return:
    """
    k_list = kwargs['k_list']
    df     = kwargs['df']
    result = []
    th = []
    local_data = MyLocal(result)
    for k_ in k_list:
        t = thrd.Thread(target=worker, args=(local_data, fitKmeanForCK,), kwargs={'k': k_, 'dataset': df})
        th.append(t)
    for t_ in th:
        t_.start()
    for i in th:
        i.join()

    return local_data.getData()

def calcMaxPricePar(**kwargs):
    """
    Create parallel env to find the cluster with the maximum average price
    :param kwargs, contain l_list:
    :return:
    """
    res = []
    th = []
    k = kwargs["k"]
    df = kwargs["df"]
    local_data = MyLocal(res)
    for cluster in range(0, k):
        t = thrd.Thread(target=worker, args=(local_data, maxAvgVariable,),
                        kwargs={'k': cluster + 1, 'dataset': df})
        th.append(t)

    for t_ in th:
        t_.start()
    for i in th:
        i.join()
    return local_data.getData()

def creatAxToPlot(data,**kwargs):
    data_ = kwargs["dataset"]
    cluster = kwargs["cluster"]
    color = kwargs["color"]

    x = data_.loc[data_['cluster'] == cluster, 'price']
    y = data_.loc[data_['cluster'] == cluster, 'speed']
    plt.scatter(x, y, color = color)

def plotDimensionsPar(**kwargs):
    """
    Create parallel env to plot
    :param kwargs, data, k_list, color_lst,and show (bool)
    :return:
    """
    import matplotlib.patches as mpatches
    res = []
    th = []
    patch = []

    data_  = kwargs["data_"]
    k_list = kwargs["k_list"]
    color  = kwargs["color_lst"]
    show   = kwargs["show"]

    local_data = MyLocal(res)
    plt.subplots(1)

    # Run for each cluster, create a thread and plot it in the result graph
    for cluster in k_list:
        patch.append(mpatches.Patch(color=color[cluster-1],  label=str(cluster) ))
        t = thrd.Thread(target=worker, args=(local_data, creatAxToPlot,),
                        kwargs={'cluster': cluster, 'dataset': data_, "color": color[cluster-1]})
        th.append(t)
    plt.legend(handles=patch)

    for t_ in th:
        t_.start()
    for i in th:
        i.join()

    if show:
        plt.show()
    return


def heatMap(centroids):
    sns.heatmap(centroids, cmap='Blues_r')

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

def run(k_to_run, log ):
    """
    This function run the program of the parallel jobs. It will apply data reading, scaling, applying kmeans algorithm
    while finding the best K, plotting 2 dimension graph and finding the cluster with higher value.
    The final time that will be displayed, is the total time for this program to run, and it does not measure a single function runtime
    :param k_to_run:
    :return:
    """
    # Read data
    log.info("Reading File to DF\n")
    start_time = time.time()
    dataset = pd.concat(read_filePar(**{"num_threads": 3, "filename": dir_path + '\computers_test.csv'}))
    dataset = dataset.iloc[:, 1::]
    df = dataset.copy()
    k = k_to_run
    df['laptop'] = df['laptop'].apply(lambda x: 1 if x == 'yes' else 0)
    df['cd'] = df['cd'].apply(lambda x: 1 if x == 'yes' else 0)

    # scale data
    log.info("Scaling the DF\n")
    df = scaleDataPar(**{"data": df})

    # Knn
    log.info("apply Kmeans Algorithm\n")
    k_list = [i + 1 for i in range(0, k)]
    k_clustering_res = applyKmeansAlgorithmPar(**{"k_list": k_list, "df": df})
    k_clustering_res.sort(key=len)
    sum_square_for_k = [np.sum(var_list) for var_list in k_clustering_res]

    #  choose best k
    opt_k = int(optimalK2(sum_square_for_k))
    plt.plot(np.arange(1, 11), sum_square_for_k)
    plt.axvline(opt_k, color='r')
    log.info(f"The optimal k according to the elbow graph - { opt_k}\n")

    # Create model with k clusters
    k = opt_k
    k_list = [i + 1 for i in range(0, k)]
    log.info(f"Creating model with {k} clusterF\n")
    cluster_df = df.copy()
    optimum_cluster = KMeans(k, cluster_df)
    dataset['cluster'] = optimum_cluster[0]

    #  Plot the data of the first k dimension
    plt.figure()
    col_lst = ['red', 'yellow', 'green', 'blue', 'orange', 'purple', 'black', 'brown', 'olive']
    plotDimensionsPar(**{"data_": dataset, "k_list": k_list, "color_lst": col_lst,"show": False})

    # Get the cluster with the max average price
    log.info("Choosing the cluster with the max average price\n")
    clusterPriceLst = calcMaxPricePar(**{"k": k, "df": dataset})
    max_price_cluster = calcMaxFromList(clusterPriceLst)
    log.info("The cluster %d has the max average price %s" % (max_price_cluster[0], round(max_price_cluster[1], 2)))
    log.info("--- Total run time of threads' programming is seconds %s---" % (time.time() - start_time))

    # plot a heatmap
    log.info("plot a heatmap\n")
    heatMap(optimum_cluster[1])
    plt.show()


def timeMeasure(log):
    log.info("The process will contain 2 parts,\n1st - reporting each function time run\n2nd - running the process")
    time.sleep(1)

    log.info("The 1nd part - reporting each function time")

    log.info("Reading data")
    firsTest = testFunction("reaDate", read_filePar, filename=dir_path + '\computers_test.csv')
    firsTest.OptMyFunParameters(40)
    firsTest.getMaxParameters()
    firsTest.getBestTime()
    for i in firsTest.test_results:
        log.info(" For %s num_threads the function runtime is %s: " % (i[2].get('num_threads'), round(i[0], 4)))

    log.info(" \nThe best num_threads is %s, with %s runtime: " % (
    firsTest.max_parameters[2].get("num_threads"), round(firsTest.best_time, 4)))
    time.sleep(2)

    dataset = pd.concat(read_filePar(**{"num_threads": 3, "filename": dir_path + '\computers_test.csv'}))
    dataset = dataset.iloc[:, 1::]
    df = dataset.copy()
    k = 10
    df['laptop'] = df['laptop'].apply(lambda x: 1 if x == 'yes' else 0)
    df['cd'] = df['cd'].apply(lambda x: 1 if x == 'yes' else 0)

    log.info("\nScaling the data")
    secondTest = testFunction("scaling_data", scaleDataPar, **{"data": df})
    secondTest.OptMyFunParameters(1)
    secondTest.getBestTime()
    log.info("The scaling data runtime is: %s " % (secondTest.best_time))
    time.sleep(1)
    df = scaleDataPar(**{"data": df})

    log.info("\nFinding the best K means for k={1, 2,..., 10}")
    k_list = [i + 1 for i in range(0, k)]
    fourthTest = testFunction("Best_price_cluster", applyKmeansAlgorithmPar, **{"k_list": k_list, "df": df})
    fourthTest.OptMyFunParameters(1)
    fourthTest.getBestTime()
    time.sleep(1)
    log.info("The applyKmeansAlgorithmPar runtime is: %s " % (fourthTest.best_time))

    log.info("\nFind cluster with the max average price\n")
    cluster_df = df.copy()
    optimum_cluster = KMeans(k, cluster_df)
    dataset['cluster'] = optimum_cluster[0]
    fifthTest = testFunction("Elbow_graph", calcMaxPricePar, **{"k": k, "df": dataset})
    fifthTest.OptMyFunParameters(1)
    fifthTest.getBestTime()
    time.sleep(1)
    log.info("The scaling data runtime is: %s " % (fifthTest.best_time))

    log.info("The 2nd part - running the process\n")
    time.sleep(1)

if __name__ == '__main__':
    log.basicConfig(level=log.DEBUG)
    log.getLogger('matplotlib.font_manager').disabled = True
    pil_logger = log.getLogger('PIL')
    pil_logger.setLevel(log.INFO)

    timeMeasure(log)
    run(10, log)



