import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def frame_level_prc_auc(frame_gts: np.ndarray, frame_scores: np.ndarray):
    # Frame-level precision-recall curve and its AUC (which is equivalent to average precision, AP)
    precision, recall, _ = precision_recall_curve(frame_gts, frame_scores, pos_label=1)
    pr_auc = auc(recall, precision)

    # Actually the same:
    # From sklearn.metrics import average_precision_score
    # pr_auc = average_precision_score(frame_gts, frame_scores)

    return pr_auc, precision, recall


def frame_level_roc_auc(frame_gts: np.ndarray, frame_scores: np.ndarray):
    # Frame-level ROC curve and its AUC
    fpr, tpr, _ = roc_curve(frame_gts, frame_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr
