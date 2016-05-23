from __future__ import absolute_import, print_function
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.io
import scipy.spatial.distance as distance
import arse.pme.multigs as multigs
import arse.pme.membership as membership
import arse.pme.homography as homography
import arse.pme.fundamental as fundamental
import arse.pme.acontrario as ac
import arse.test.utils as test_utils
import arse.test.test_transformations as test_transformations


def load(path, tol=1e-5):
    data = scipy.io.loadmat(path)

    x = data['data'].T
    gt = np.squeeze(data['label'])

    # remove repeated points
    m = x.shape[0]
    dist = distance.squareform(distance.pdist(x)) + np.triu(np.ones((m, m)), 0)
    mask = np.all(dist >= tol, axis=1)
    gt = gt[mask]
    x = x[mask, :]

    # sort in reverse order (inliers first, outliers last)
    inv_order = np.argsort(gt)[::-1]
    gt = gt[inv_order]
    x = x[inv_order, :]

    x[:, 0:2] -= np.array(data['img1'].shape[:2], dtype=np.float) / 2
    x[:, 3:5] -= np.array(data['img2'].shape[:2], dtype=np.float) / 2

    data['data'] = x
    data['label'] = gt
    return data


def run(transformation, inliers_threshold, regular_nfa=True):
    log_filename = 'logs/test_{0}_{1}'.format(transformation, inliers_threshold)
    if regular_nfa:
        log_filename += '_regular_nfa'
    print(log_filename)
    logger = test_utils.Logger(log_filename + '.txt')
    sys.stdout = logger

    n_samples = 2000
    epsilon = 0

    path = '../data/adelaidermf/{0}/'.format(transformation)

    filenames = []
    for (_, _, fn) in os.walk(path):
        filenames.extend(fn)
        break

    stats_list = []
    for example in filenames:
        data = load(path + example)

        if transformation == 'homography':
            model_class = homography.Homography
            if not regular_nfa:
                img_size = data['img2'].shape[:2]
                nfa_proba = np.pi / np.prod(img_size)
        else:
            model_class = fundamental.Fundamental
            if not regular_nfa:
                img_size = data['img2'].shape[:2]
                nfa_proba = (2. * np.linalg.norm(img_size) / np.prod(img_size))

        generator = multigs.ModelGenerator(model_class, data['data'], n_samples)
        min_sample_size = model_class().min_sample_size
        if regular_nfa:
            local_ratio = 3.
            proba = 1. / local_ratio
            tester = ac.BinomialNFA(epsilon, proba, min_sample_size)
            thresholder = membership.LocalThresholder(inliers_threshold,
                                                      ratio=local_ratio)
        else:
            tester = ac.ImageTransformNFA(epsilon, nfa_proba, min_sample_size)
            thresholder = membership.GlobalThresholder(inliers_threshold)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        prefix = example[:-4]
        dir_name = '{0}_{1:e}'.format(transformation, inliers_threshold)

        res = test_transformations.test(model_class, data, prefix, generator,
                                        thresholder, tester, dir_name)
        stats_list.append(res)

        print('-'*40)
        plt.close('all')

    reg_list, comp_list = zip(*stats_list)

    print('Statistics of regular bi-clustering')
    test_utils.compute_stats(reg_list)
    print('Statistics of compressed bi-clustering')
    test_utils.compute_stats(comp_list)
    print('-'*40)

    sys.stdout = logger.stdout
    logger.close()


def run_all():
    for thresh in np.arange(1, 10, .5):
        run('homography', thresh, regular_nfa=False)
    for thresh in np.arange(1, 20, .5):
        run('fundamental', thresh, regular_nfa=False)
    run('homography', 14.5, regular_nfa=True)
    run('fundamental', 3., regular_nfa=True)

if __name__ == '__main__':
    run_all()
    plt.show()