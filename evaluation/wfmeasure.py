import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter


def MultiClassWFmeasure(FM, GT):
    unique_classes = np.unique(GT)
    class_scores = []
    class_indexes = []
    for i in unique_classes:
        class_FM = FM == i
        class_GT = GT == i
        class_score = weighted_f_beta_score(class_FM, class_GT)
        class_scores.append(class_score)
        class_indexes.append(int(i))

    return class_scores, class_indexes


def weighted_f_beta_score(candidate, gt, beta=1.0):
    candidate = candidate.astype(np.float64)

    if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
        raise ValueError("'candidate' values must be inside range [0 - 1]")

    if gt.dtype in [bool]:
        gt_mask = gt
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.array(gt, dtype=candidate.dtype)
    else:
        if not np.all(np.isclose(gt, 0) | np.isclose(gt, 1)):
            raise ValueError("'gt' must be a 0/1 or boolean array")
        gt_mask = np.isclose(gt, 1)
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.asarray(gt, dtype=candidate.dtype)

    E = np.abs(candidate - gt)
    dist, idx = distance_transform_edt(not_gt_mask, return_indices=True)

    Et = np.array(E)
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
    sigma = 5.0
    EA = gaussian_filter(Et, sigma=sigma, truncate=3 / sigma, mode="constant", cval=0.0)
    min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

    B = np.ones(gt.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
    Ew = min_E_EA * B

    eps = np.spacing(1)
    TPw = np.sum(gt) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    R = 1 - np.mean(Ew[gt_mask])
    P = TPw / (eps + TPw + FPw)
    Q = (1 + beta**2) * (R * P) / (eps + R + (beta * P))
    return Q
