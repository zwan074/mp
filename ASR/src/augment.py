import random
import numpy as np
import torch



def augment_list():  #  oeprations and their ranges
    l = [
        (freqMask_one, 0.0, 0.15),  # 0
        (timeMask_one, 0.0, 0.20),  # 1
    ]
    return l

aug_list = [
        ['freqMask_one', '0.0', '0.15'],  # 0
        ['timeMask_one', '0.0', '0.20'],  # 1
    ]

#augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

print(aug_list)

def get_augment(name):
    return augment_dict[name]


def apply_augment(x, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(x, level * (high - low) + low)

def policy_decoder(augment, num_policy, num_op):
    op_list = augment_list(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies
