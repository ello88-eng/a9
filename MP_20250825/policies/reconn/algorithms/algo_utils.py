import random

import numpy as np
from scipy.stats import variation


def cal_obj(X, chm, cluster_size: int, ncluster: int):
    obj = [None] * len(chm)
    for i in range(len(chm)):
        idx_cnt = 0
        tmp_obj = 0
        for _ in range(ncluster):
            tmp = []
            for _ in range(cluster_size):
                try:
                    tmp.append(X[chm[i][idx_cnt]])
                except:
                    print(f"tmp:{tmp} \t X:{X} \t chm:{chm} \t i:{i} \t idx_cnt:{idx_cnt}")
                    continue
                idx_cnt += 1
            axis_var = variation(tmp, axis=0)
            total_var = variation(axis_var)
            tmp_obj += total_var
        obj[i] = tmp_obj
    return obj


def same_size_clustering(X, cluster_size: int, ncluster: int, labels, num_iters: int = 1000):
    # Genetic Algorithm (manually written)
    # [Step 1] Generate Initial Chromosome based on Kmeans algorithm ---
    # final output
    chm = []
    # global objective value
    g_obj = np.inf
    # local chormosome (10 chromosomes are set as an one population/generation)
    l_chm = [None] * 10
    # random chromosome generation
    for i in range(9):
        l_chm[i] = np.random.permutation([i for i in range(cluster_size * ncluster)]).tolist()
    l_chm[9] = []
    # Applying K-means result
    for j in range(ncluster):
        l_chm[9].extend(np.where(labels == j)[0].tolist())
    # [Step 2] Generate Initial Chromosome based on Kmeans algorithm ---
    null_trial = 0
    for _ in range(num_iters):
        # (step 2-1) calculate objective value (variance of the each cluster)
        # local obj value
        l_obj = cal_obj(X, l_chm, cluster_size, ncluster)
        # (step 2-2) update global objective value & chromosome
        if min(l_obj) < g_obj:
            g_obj = min(l_obj)
            chm = l_chm[np.where(l_obj == min(l_obj))[0][0]]
            null_trial = 0
        else:
            null_trial += 1
            l_obj[0] = g_obj
            l_chm[0] = chm
        # if objective value doesn't anymore continuosly--> break
        if null_trial > 50:
            break
        # (step 2-3) crossover (one-point crossver)
        sort_list = np.argsort(l_obj)[0:5]
        # local chormosome (10 chromosomes are set as an one population/generation)
        ll_chm = [None] * 10
        for p3 in range(0, len(sort_list)):
            cut_ratio = random.randint(int(0.3 * len(X)), int(0.7 * len(X)))
            ll_chm[2 * p3] = l_chm[sort_list[p3]][cut_ratio + 1 :] + l_chm[sort_list[p3]][0 : cut_ratio + 1]
            ll_chm[2 * p3 + 1] = l_chm[sort_list[p3]][0 : cut_ratio + 1] + list(
                reversed(l_chm[sort_list[p3]][cut_ratio + 1 :])
            )
        l_chm = ll_chm
    # [Step 3] Cluter labeling & center positionpythonscipy
    label = [int(x) for x in range(len(chm))]
    for j in range(len(chm)):
        label[chm[j]] = j // cluster_size
    center = []
    for j in range(ncluster):
        set = np.where(np.array(label) == j)
        x_ = [X[set[0][k]][0] for k in range(cluster_size)]
        y_ = [X[set[0][k]][1] for k in range(cluster_size)]
        z_ = [X[set[0][k]][2] for k in range(cluster_size)]
        center.append([sum(x_) / cluster_size, sum(y_) / cluster_size, sum(z_) / cluster_size])
    return label, center, g_obj
