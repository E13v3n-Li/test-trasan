#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import copy
import torch
from torch import nn


def FedAvg(w, weights=None):
    if not w:
        raise ValueError('Cannot aggregate an empty model list')
    if weights is None:
        weights = [1.] * len(w)
    if len(weights) != len(w) or any(x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError('Invalid aggregation weights')
    weights = [x / sum(weights) for x in weights]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if w_avg[k].is_floating_point():
            w_avg[k] = sum(state[k] * weight for state, weight in zip(w, weights))
        else:
            # Integer counters (e.g. BN num_batches_tracked) must retain their dtype.
            w_avg[k] = torch.stack([state[k] for state in w]).max(dim=0).values
    return w_avg
