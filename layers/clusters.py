"""
@AikenH 2021 6.18 CLUSTER and DownSampler
Intergrate Cluster in this files
using the sklearn to help us build this, but we should figure out
how sklearn deal with those data which contain batchsize.
Scikit-Learn : CLUSTER https://scikit-learn.org/stable/modules/clustering.html
PCA:  https://www.cnblogs.com/pinard/p/6243025.html
TSNE: https://www.deeplearn.me/2137.html
        https://www.cntofu.com/book/170/docs/71.md
CUML and CUPY
Why tene donot have fit: https://stackoverflow.com/questions/59214232/python-tsne-transform-does-not-exist
"""
import torch
import copy
import numpy as np
import cupy as cp
from torch import nn
from scipy.optimize import linear_sum_assignment

class makeCluster():
    def __init__(self, cluster_T:str, iscuda=True, **kwargs):
        self.cluster_T = cluster_T
        self.kwargs = kwargs
        self.iscuda = iscuda
        self.clus = self.GetCluster(cluster_T,**kwargs)

    def fit(self,X):
        self.clus.fit(X)
        return self.clus

    def predict(self,X):
        return self.clus.predict(X)

    def cluster_centers_(self):
        return self.clus.cluster_centers_

    def labels_(self):
        return self.clus.labels_

    def GetCluster(self,cluster_t, **kwargs):
        # kwargs 按照我们需要的聚类类型来输入参数。
        if cluster_t.lower() == 'kmeans':
            # kmeans++用来初始化kmean的起始点
            if not self.iscuda:
                from sklearn.cluster import KMeans
            else:
                from cuml import KMeans
            clus = KMeans(**kwargs)

        else:
            raise NotImplementedError('GetCluster not supposed this right now')

        return clus

class makeDownsample():
    def __init__(self, down_t:str, iscuda=True, **kwargs):
        self.down_t = down_t
        self.kwargs = kwargs
        self.iscuda = iscuda
        self.sampler = self.GetDownsample(down_t, **kwargs)

        return None

    def GetDownsample(self, down_t, **kwargs):
        if down_t.lower() == 'tsne':
            if not self.iscuda:
                from sklearn.manifold import TSNE
            else:
                from cuml import TSNE

            sampler = TSNE(**kwargs)

        elif down_t.lower() == 'pca':
            if not self.iscuda:
                from sklearn.decomposition import PCA
            else:
                from cuml import PCA

            sampler = PCA(**kwargs)

        return sampler

    def fit_transform(self,X):
        return self.sampler.fit_transform(X)

    def fit(self, X):
        return self.sampler.fit(X)

    def transform(self, X):
        retruns = self.sampler.transform(X)

class LabelGenerator(nn.Module):
    def __init__(self, cluster_t=None, cluster_dict=None, downsample_t=None ,downsample_dict=None,
                projector=None, iscuda=True,*args, **kwargs):
        super(LabelGenerator, self).__init__()
        # generate the component
        self.projector = projector
        self.iscuda = iscuda
        self.downsampler = makeDownsample(downsample_t, iscuda=iscuda,**downsample_dict)
        self.cluster = makeCluster(cluster_t, iscuda=iscuda, **cluster_dict)
        self.map_dict = None
        self.o_cluster_center = None

        # recheck those component. we only allow one of  projectors or downsampler
        if self.projector is not None and self.downsampler is not None:
            self.downsampler = None
            print("Warning: downsampler is not allowed to be used with projector")
            print("so we make downsampler None")
        return None

    def forward(self, x):
        """
        # !! change the 80 as a params 
        accept a feature or a logits, return a 
        """
        assert self.map_dict is not None, "u need to init the labelGe first"

        if self.projector is not None:
            if x.dim() == 1: x.unsqueeze_(0)
            x = self.projector(x).detach()

        if not self.iscuda: val = x.data.cpu().numpy()
        else: val = copy.deepcopy(x.data).detach()

        if val.ndim == 1:
            val = np.expand_dims(val, axis=0) if not self.iscuda else val.unsqueeze(0)

        # we using downsampler or projectors here
        if self.downsampler is not None:
            val = self.downsampler.transform(val)

        # using cluster to get the real label
        label = self.cluster.predict(val)
        # align with the true label
        targets = self.map_dict[label.item()]
        return targets

    def align_gt(self, pred, labels, topk=3):
        """Align the preditions with the true labels,
        which should be carry out befor the final test
        rather the init.

        Args:
            pred (): the predition of new datas.
            labels (): the GT label of those datas
            topk (int, optional): chose topn labels in it.

        Returns:
            None, generate the mapping dict instead.
        """

        num_cls, num_new_cls = len(cp.unique(labels)), len(cp.unique(pred))
        cost = cp.zeros((num_new_cls,num_cls))

        list_p2r_class, topk_appear = [], []
        for i in range(num_cls):
            # 0. build a array for each pseudo class's real labels(pseudo to real)
            list_p2r_class.append(
                [labels[idx] for idx in range(len(pred)) if pred[idx] == i]
            )
            num_this = len(list_p2r_class[i])

            # 1. calculate the topk of each pseudo label
            if isinstance(list_p2r_class[i][0],torch.Tensor):
                list_p2r_class[i] = torch.tensor(list_p2r_class[i])
            list_p2r_class[i] = cp.array(list_p2r_class[i])
            list_p2r_class[i] = cp.bincount(list_p2r_class[i])
            # we may not need the topk, we can use the nums of data to build the cost matrix for this
            # topk_appear.append(cp.argpartition(list_p2r_class[i], -topk)[-topk:])

            # 2. build the cost matrix
            # cost[i] = [list_p2r_class[i][idx] for idx in topk_appear[i]]
            for j in range(80, len(list_p2r_class[i])):
                cost[i][j-80] = float(list_p2r_class[i][j]) / float(num_this)

        cost = -cost # change the weight to the cost

        # 3. using the KM algorithm to calculate get the relationship
        # between the pseudo class and the real class
        if isinstance(cost, cp._core.core.ndarray):
            cost = cp.asnumpy(cost)
        key, value = linear_sum_assignment(cost)

        # 4, regiest the map
        mapping_dict,self.reverse_map = {},{}
        for k,v in zip(key, value):
            # mapping_dict[k] = v + 80
            # whether the k here need to add 80
            self.reverse_map[v+80] = self.map_dict[k]

        assert len(np.unique(key)) == len(np.unique(value)), "we need a one on one result, or we will need to rebuild this"
        return mapping_dict

    def align_old(self, new_clus_center):
        """
        We align the new_clus with the old, so we need a mapping dict.
        if the without mapping dict, we will init the original one like a->a.

        others, we using the distance of cluster_center to get the cost, 
        and using km to get the response true.
        """
        # init the mapping
        if self.map_dict is None:
            new_map = {}
            for i in range(20):
                new_map[i] = i+80
            return new_map

        # align the cluser with the old one
        assert self.o_cluster_center is not None, "u need to init the labelGe first"

        cost = []
        for i in range(len(self.o_cluster_center)):
            cost.append([cp.linalg.norm(new_clus_center[i] - old_center)
                            for old_center in self.o_cluster_center])
        cost = cp.array(cost)

        if isinstance(cost, cp._core.core.ndarray): cost = cp.asnumpy(cost)
        key, value = linear_sum_assignment(cost)

        # regited the result to the mapping dict
        new_map = {}
        for k,v in zip(key, value):
            new_map[k] = self.map_dict[v]

        return new_map

    def update(self, features, labels=None, skip_down=False):
        """
        1. save the feature or the logits in the test process to update the LabelGe
        2. Using GT and pred to carry out the mapping 
        """
        # get the bak of cluster and generate the new
        self.o_cluster_center = copy.deepcopy(self.cluster.cluster_centers_)

        # get the downsamper features (if we use logits, we donot downsample it）
        if not skip_down and self.downsampler is not None:
            self.downsampler = self.downsampler.fit(features)
            dn_features = self.downsampler.transform(features)
        else:
            if not self.iscuda:
                features = torch.tensor(features)

            dn_features = self.projector(
                features).detach() if self.projector is not None else features

            if not self.iscuda:
                dn_features = dn_features.numpy()

        self.cluster = self.cluster.fit(dn_features)

        # align the old cluster
        if labels is None: self.map_dict = self.align_old(self.cluster.cluster_centers_)
        # align labels with GT
        else: self.map_dict = self.align_gt(self.cluster.labels_, labels)

        return True

if __name__ == "__main__":
    # create a sample to test the cluster and the downsampler test the result of it
    # using example from sklearn to test it
    features = np.random.rand(20,10)
    labels = np.random.randint(0,3,size=20)
    print("befor cluster those label will be like : \n {} \n".format(labels))

    # get the pca method and calculate it, the tsne cannot used in the test, so it only suppost fit_transform
    sampler = makeDownsample('pca', n_components=2,)
    sampler = sampler.fit(features)
    new_feature = sampler.transform(features)
    # print(new_feature.shape)

    # using cluster to clust those data
    clus = makeCluster('kmeans', n_clusters=3, init='k-means++', random_state=0)
    clus = clus.fit(new_feature)
    print("after cluster those label will be like : \n {} \n".format(clus.labels_))

    # get the important function for the cluster and the downsampler
    try_one = np.random.rand(1,10)
    try_one = sampler.transform(try_one)
    new_label = clus.predict(try_one)
    print("feature: {} \n label : {} \n".format(try_one, new_label))

    # visulize the result of the cluser can reference to the tsne visulization
    idx = [np.where(clus.labels_ == i) for i in range(3)]
    res = [labels[idx[i]] for i in range(3)]

    print(idx)
    print(res)

    # using the label which appear most times in one cluster as the true label
    # then may cause one labels append twich, which inplace we will using the second labeld
    # using the method which get the most k nums of it, then we dicide it.
    newidx = [np.argmax(np.bincount(res[i])) for i in range(3)]
    print(newidx)
