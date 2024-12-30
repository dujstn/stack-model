import torch
import numpy as np
import cv2 as cv


def updateConfusion(pred: dict[torch.tensor], truth: dict, confidence: float, confusion: np.ndarray) -> None:
    """
    Updates the confusion matrix with the intersection over union data for the current prediction and ground
    truth mask.

    pred contains attributes 'boxes', 'labels', 'scores', and 'masks', corresponding to the mask of each
    predicted class and its associated confidence score.

    truth contains attributes 'boxes', 'labels', and 'labels', corresponding to the ground truth masks of the
    same image.
    """
    # Filter out duplicate predictions and predictions under the provided confidence value
    p_msk = pred[0]['masks'][:, 0].detach().cpu().numpy()
    p_scr = pred[0]['scores'][:].detach().cpu().numpy()
    p_lab = pred[0]['labels'][:].detach().cpu().numpy()
    t_msk = truth['masks'].numpy()
    t_lab = truth['labels'].numpy()

    classes_seen = set()
    fil = np.array([False] * p_lab.size, dtype=bool)
    for i in range(p_lab.size):
        if p_lab[i] not in classes_seen and p_scr[i] >= confidence:
            classes_seen.add(p_lab[i])
            fil[i] = True
        else:
            fil[i] = False
    p_msk, p_lab, = p_msk[fil], p_lab[fil]
    p_msk, t_msk = (p_msk[:, :, :] > confidence).astype(bool), t_msk.astype(bool)

    pred_ref = {p_lab[i]: p_msk[i] for i in range(len(p_lab))}
    truth_ref = {t_lab[i]: t_msk[i] for i in range(len(t_lab))}

    # Iterate through each class and update the confusion matrix
    labs = set(np.concatenate((p_lab, t_lab)))
    for lab in labs:
        if lab in pred_ref and lab in truth_ref:
            tp = np.sum(np.logical_and(pred_ref[lab], truth_ref[lab]))
            tp_zeroes = np.sum(np.logical_and(np.logical_not(pred_ref[lab]), np.logical_not(truth_ref[lab])))
            fp = np.sum(np.logical_and(
                pred_ref[lab], np.logical_not(truth_ref[lab])))
            fn = np.sum(np.logical_and(
                np.logical_not(pred_ref[lab]), truth_ref[lab]))

            confusion[0][0] += tp_zeroes
            confusion[lab - 1][lab - 1] += tp
            confusion[0][lab - 1] += fn
            confusion[lab - 1][0] += fp

        elif lab not in truth_ref:
            tp = np.sum(np.logical_not(pred_ref[lab]))
            fp = np.sum(pred_ref[lab])

            confusion[0][0] += tp
            confusion[lab - 1][0] += fp

        else:
            tp = np.sum(np.logical_not(truth_ref[lab]))
            fn = np.sum(truth_ref[lab])

            confusion[0][lab - 1] += fn
            confusion[0][0] += tp


def getmIOU(confusion: np.ndarray):
    class_scores = np.zeros(len(confusion))

    for i in range(len(confusion)):
        tp = confusion[i][i]
        fp = np.sum(confusion[:][i]) - tp
        fn = np.sum(confusion[i][:]) - tp

        class_scores[i] = tp / (tp + fp + fn)
    class_scores[np.isnan(class_scores)] = 1
    print(class_scores)

    return np.average(class_scores)
