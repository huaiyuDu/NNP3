"""Normalize features"""

import numpy as np


def normalize(features, features_mean=None, features_deviation=None):

    features_normalized = np.copy(features).astype(float)
    if features_mean is None:
        features_mean = np.mean(features, 0)
    # 计算均值


    # 计算标准差
    if features_deviation is None:
        features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


def denormalize(features, features_mean=None, features_deviation=None):

    features_denormalized = np.copy(features).astype(float)

    features_deviation[features_deviation == 0] = 1
    features_denormalized *=features_deviation
    features_denormalized = features_denormalized + features_mean
    return features_denormalized

def normalize_minus_one_to_one(features, features_mean=None, range_width=None):

    features_normalized = np.copy(features).astype(float)
    if features_mean is None:
        features_mean = np.mean(features, 0)
    # 计算均值


    # 计算标准差
    if range_width is None:
        range_width = np.abs(features.min(axis=0, keepdims=True) - features.max(axis=0, keepdims=True))*2

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    range_width[range_width == 0] = 1
    features_normalized /= range_width

    return features_normalized, features_mean, range_width

def normalize_zero_to_one(features,features_min= None,range_width=None):

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    if features_min is None:
        features_min = features.min(axis=0, keepdims=True)

    # 计算标准差
    if range_width is None:
        range_width = np.abs(features.min(axis=0, keepdims=True) - features.max(axis=0, keepdims=True))
    features_normalized -= features_min
    # 标准化操作
    # if features.shape[0] > 1:
    #     features_normalized -= features_mean

    # 防止除以0
    range_width[range_width == 0] = 1
    features_normalized /= range_width

    return features_normalized, features_min,range_width
# a = np.array([[1,2], [2,4],[3,6]])
# print(normalize(a))
# (normal, mean ,deviation) = normalize(a)
# print(denormalize(normal,mean,deviation))