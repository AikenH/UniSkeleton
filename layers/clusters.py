"""
@AikenH 2021 6.18 CLUSTER 
# Intergrate Cluster in this files
# using the sklearn to help us build this, but we should figure out
# how sklearn deal with those data which contain batchsize.
# Scikit-Learn : CLUSTER https://scikit-learn.org/stable/modules/clustering.html
"""

import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np


def GetCluster(cluster_T:str,**kwargs):
    # kwargs 按照我们需要的聚类类型来输入参数。
    if cluster_T.lower() == 'kmeans':
        # kmeans++用来初始化kmean的起始点
        clus = cluster.KMeans(**kwargs)
    
    else:
        raise NotImplementedError('GetCluster not supposed this right now')
    return clus


# test the distribution of those cluster methods
# visualize the data and the clust result by matplotlib
# try create those data with different batch size (or we just concate all the date to the cluster)
if __name__ == "__main__":
    # setting the cluster params
    isVisulize = True
    params = {
        'n_clusters':2,
        'random_state':0,
        'init':'k-means++'
    }

    # set up those data by sklearn
    if isVisulize:
        # official without visulize
        X = np.array([[1,2],[1,4],[1,0],
                    [10,2],[10,4],[10,0]])
        res = GetCluster('kmeans',**params)
        res.fit(X)
        print(res.labels_)
        print(res.predict([[12,3],[0,0]]))
        print(res.cluster_centers_)
    
    else:
        
        ... 
    


