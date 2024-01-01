import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def frame_level_prc_auc(frame_gts: np.ndarray, frame_scores: np.ndarray, num_thresholds: int = 10000):
    # Frame-level precision-recall curve and its AUC (which is equivalent to average precision, AP)
    precision, recall, _ = precision_recall_curve(frame_gts, frame_scores, pos_label=1)
    pr_auc = auc(recall, precision)

    # Sample num_thresholds points for plotting
    if len(precision) > num_thresholds:
        sampled_indices = np.linspace(0, len(precision) - 1, num_thresholds, dtype=np.int32)
        precision = precision[sampled_indices]
        recall = recall[sampled_indices]

    return pr_auc, precision, recall


def frame_level_roc_auc(frame_gts: np.ndarray, frame_scores: np.ndarray, num_thresholds: int = 10000):
    # Frame-level ROC curve and its AUC
    fpr, tpr, _ = roc_curve(frame_gts, frame_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Sample num_thresholds points for plotting
    if len(fpr) > num_thresholds:
        sampled_indices = np.linspace(0, len(fpr) - 1, num_thresholds, dtype=np.int32)
        fpr = fpr[sampled_indices]
        tpr = tpr[sampled_indices]

    return roc_auc, fpr, tpr
