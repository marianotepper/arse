from __future__ import absolute_import
import numpy as np
import itertools as itt
import arse.biclustering.utils as bic_utils
import arse.pme.acontrario as ac


def clean(model_class, x, thresholder, ac_tester, bic_list,
          check_overlap=False, share_elements=True):
    min_sample_size = model_class().min_sample_size
    bic_list = [bic for bic in bic_list
                if bic[1].nnz > 1 and bic[0].nnz >= min_sample_size]

    inliers_list = []
    model_list = []
    for lf, _ in bic_list:
        inliers = np.squeeze(lf.toarray())
        mod = model_class(x[inliers])
        model_list.append(mod)
        inliers = thresholder.membership(mod, x)
        inliers_list.append(inliers)

    if not model_list:
        return [], []

    # filter out non-meaningful groups
    keep = _meaningful(ac_tester, inliers_list)
    inliers_list, model_list, bic_list = _filter_in(keep, inliers_list,
                                                    model_list, bic_list)

    keep = ac.exclusion_principle(x, thresholder, ac_tester, inliers_list,
                                  model_list)
    inliers_list, model_list, bic_list = _filter_in(keep, inliers_list,
                                                    model_list, bic_list)

    if check_overlap:
        keep = _keep_disjoint(ac_tester, inliers_list)
        inliers_list, model_list, bic_list = _filter_in(keep, inliers_list,
                                                        model_list, bic_list)

    if not share_elements and inliers_list:
        _solve_intersections(x, inliers_list, model_list)
        keep = _meaningful(ac_tester, inliers_list)
        inliers_list, model_list, bic_list = _filter_in(keep, inliers_list,
                                                        model_list, bic_list)

    bic_list = _inliers_to_left_factors(inliers_list, bic_list)

    return model_list, bic_list


def _meaningful(ac_tester, inliers_list):
    keep = filter(lambda e: ac_tester.meaningful(e[1]), enumerate(inliers_list))
    if keep:
        return zip(*keep)[0]
    else:
        return []


def _filter_in(keep, inliers_list, model_list, bic_list):
    inliers_list = [inliers_list[s] for s in keep]
    model_list = [model_list[s] for s in keep]
    bic_list = [bic_list[s] for s in keep]
    return inliers_list, model_list, bic_list


def _keep_disjoint(tester, inliers_list, tol=0.3):
    size = range(len(inliers_list))
    to_remove = []
    for i1, i2 in itt.combinations(size, 2):
        in1 = inliers_list[i1]
        in2 = inliers_list[i2]
        in1_binary = in1 > 0
        in2_binary = in2 > 0
        overlap = float(np.sum(np.logical_and(in1_binary, in2_binary)))
        overlap /= np.maximum(np.sum(in1_binary), np.sum(in2_binary))
        if overlap > tol:
            if tester.nfa(in1) < tester.nfa(in2):
                to_remove.append(i2)
            else:
                to_remove.append(i1)
    keep = set(size) - set(to_remove)
    return keep


def _solve_intersections(x, inliers_list, model_list):
    intersection = np.sum(np.vstack(inliers_list) > 0, axis=0) > 1
    dists = [mod.distances(x[intersection, :]) for mod in model_list]
    idx = np.argmin(np.vstack(dists), axis=0)
    for i, inliers in enumerate(inliers_list):
        inliers[intersection] = idx == i


def _inliers_to_left_factors(inliers_list, bic_list):
    left_factors = [bic_utils.sparse(inliers[:, np.newaxis] > 0,
                                     dtype=np.bool)
                    for inliers in inliers_list]
    return [(lf, rf) for lf, (_, rf) in zip(left_factors, bic_list)]
