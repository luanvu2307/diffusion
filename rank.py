#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print


def search(queries, gallery, truncation_size, kd, cache_dir, gnd_path):
    n_query = len(queries)
    diffusion = Diffusion(np.vstack([queries, gallery]), cache_dir)
    offline = diffusion.get_offline_results(truncation_size, kd)
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    scores = features[:n_query] @ features[n_query:].T
    ranks = np.argsort(-scores.todense())
    evaluate(ranks, gnd_path)


def search_old(queries, gallery, truncation_size, kd, kq, cache_dir, gnd_path, gamma=3):
    diffusion = Diffusion(gallery, cache_dir)
    offline = diffusion.get_offline_results(truncation_size, kd)

    time0 = time.time()
    print('[search] 1) k-NN search')
    sims, ids = diffusion.knn.search(queries, kq)
    sims = sims ** gamma
    qr_num = ids.shape[0]

    print('[search] 2) linear combination')
    all_scores = np.empty((qr_num, truncation_size), dtype=np.float32)
    all_ranks = np.empty((qr_num, truncation_size), dtype=np.int64)
    for i in tqdm(range(qr_num), desc='[search] query'):
        scores = sims[i] @ offline[ids[i]]
        parts = np.argpartition(-scores, truncation_size)[:truncation_size]
        ranks = np.argsort(-scores[parts])
        all_scores[i] = scores[parts][ranks]
        all_ranks[i] = parts[ranks]
    print('[search] search costs {:.2f}s'.format(time.time() - time0))

    # 3) evaluation
    evaluate(all_ranks, gnd_path)


def evaluate(ranks, gnd_path):
    gnd_name = os.path.splitext(os.path.basename(gnd_path))[0]
    with open(gnd_path, 'rb') as f:
        gnd = pickle.load(f)['gnd']
    compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)


def search_unseen_query(queries, gallery, truncation_size, kd, kq, cache_dir, gnd_path, gamma=3):
    """
    Search unseen query
    """

    # dataset = Dataset(query_path, gallery_path)
    if not os.path.isdir(cache_dir):
      os.makedirs(cache_dir)

    diffusion = Diffusion(gallery, cache_dir)
    offline = diffusion.get_offline_results(truncation_size, kd)

    sims, ids = diffusion.knn.search(queries, kq)
    sims = sims ** gamma
    qr_num = ids.shape[0]

    all_scores = np.empty((qr_num, truncation_size), dtype=np.float32)
    all_ranks = np.empty((qr_num, truncation_size), dtype=np.int64)
    for i in tqdm(range(qr_num), desc='[search] query'):
        scores = sims[i] @ offline[ids[i]]
        parts = np.argpartition(-scores, truncation_size)[:truncation_size]
        ranks = np.argsort(-scores[parts])
        all_scores[i] = scores[parts][ranks]
        all_ranks[i] = parts[ranks]

    return list(all_ranks[0][:10])
