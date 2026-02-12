from sklearn.preprocessing import MinMaxScaler
import numpy as np
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.evaluation.basic_metrics import basic_metricor
import pandas as pd


def normalize_score(score, train_index):

    scaler = MinMaxScaler(feature_range=(0, 1))

    normalized_score = np.empty_like(score)
    normalized_score[:train_index] = scaler.fit_transform(score[:train_index].reshape(-1, 1)).ravel()
    normalized_score[train_index:] = scaler.transform(score[train_index:].reshape(-1, 1)).ravel()
    
    return normalized_score

def compute_AUC_PR(score_df, labels, file):
    grader = basic_metricor()
    results_AUC_PR = {}
    results_AUC_PR['file'] = file
    for AD_name in score_df.columns:
        results_AUC_PR[AD_name] = grader.metric_PR(labels, score_df[AD_name].to_numpy())

    metric_AUC_PR_df = pd.DataFrame([results_AUC_PR])
    return metric_AUC_PR_df

def normalize_minmax_score(score):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_score = scaler.fit_transform(score.reshape(-1, 1)).ravel()

    return normalized_score

