from __future__ import absolute_import, print_function
import sys
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import scipy.sparse as sp
import numpy as np
import scipy.io
import re
import timeit
import arse.biclustering as bc
import arse.test.utils as test_utils
import arse.pme.preference as pref
import arse.pme.sampling as sampling
import arse.pme.line as line
import arse.pme.circle as circle
import arse.pme.acontrario as ac


def base_plot(x):
    x_lim = (x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
    y_lim = (x[:, 1].min() - 0.1, x[:, 1].max() + 0.1)
    delta_x = x_lim[1] - x_lim[0]
    delta_y = y_lim[1] - y_lim[0]
    min_delta = min([delta_x, delta_y])
    delta_x /= min_delta
    delta_y /= min_delta
    fig_size = (4 * delta_x, 4 * delta_y)

    plt.figure(figsize=fig_size)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.scatter(x[:, 0], x[:, 1], c='w', marker='o', s=10)


def plot_models(models, palette=None, **kwargs):
    if palette is not None and 'color' in kwargs:
        raise RuntimeError('Cannot specify palette and color simultaneously.')
    for i, mod in enumerate(models):
        if palette is not None:
            kwargs['color'] = palette[i]
        mod.plot(**kwargs)


def plot_final_models(x, models, palette):
    base_plot(x)
    plot_models(models, palette=palette, linewidth=5, alpha=0.5)


def plot_original_models(x, original_models, right_factors, palette):
    base_plot(x)
    for i, rf in enumerate(right_factors):
        plot_models([original_models[j] for j in sp.find(rf)[1]],
                    color=palette[i], alpha=0.5)


def ground_truth(model_class, data, threshold, n_groups, group_size=50):
    gt_groups = []
    for i in range(n_groups):
        g = np.zeros((len(data),), dtype=bool)
        g[i * group_size:(i+1) * group_size] = True
        model = model_class(data=data[g])
        inliers = np.abs(model.distances(data)) <= threshold
        gt_groups.append(inliers)
    return gt_groups


def run_biclustering(model_class, x, original_models, pref_matrix, comp_level,
                     ac_tester, gt_groups, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    models, bic_list = test_utils.clean(model_class, x, ac_tester, bic_list)

    palette = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    plot_final_models(x, models, palette=palette)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    plot_original_models(x, original_models, [bic[1] for bic in bic_list],
                         palette)
    plt.savefig(output_prefix + '_bundles.pdf', dpi=600)

    bc_groups = [bic[0] for bic in bic_list]
    gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)

    return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)


def test(model_class, x, name, ransac_gen, ac_tester, gt_groups):
    print(name, x.shape)

    output_prefix = '../results/' + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = pref.build_preference_matrix(x.shape[0],
                                                            ransac_gen,
                                                            ac_tester)
    print('Preference matrix size:', pref_matrix.shape)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix})

    base_plot(x)
    plot_models(orig_models, alpha=0.2)
    plt.savefig(output_prefix + '_original_models.pdf', dpi=600)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    compression_level = None
    stats_reg = run_biclustering(model_class, x, orig_models, pref_matrix,
                                 compression_level, ac_tester, gt_groups,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 32
    stats_comp = run_biclustering(model_class, x, orig_models, pref_matrix,
                                  compression_level, ac_tester, gt_groups,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp


def run_all(oversampling=1):
    logger = test_utils.Logger("test_2d.txt")
    sys.stdout = logger

    inliers_threshold = 0.015
    epsilon = 0

    config = {'Star': (line.Line, sampling.AdaptiveSampler(),
                       ac.LocalNFA, oversampling * 20),
              'Stairs': (line.Line, sampling.AdaptiveSampler(),
                         ac.LocalNFA, oversampling * 20),
              'Circles': (circle.Circle, sampling.AdaptiveSampler(),
                          ac.circle.LocalNFA, oversampling * 40),
              }

    stats_list = []
    mat = scipy.io.loadmat('../data/JLinkageExamples.mat')
    for example in mat.keys():
        for c in config:
            if example.find(c) == 0:
                ex_type = c
                break
        else:
            continue

        model_class, sampler, tester_class, sampling_factor = config[ex_type]
        data = mat[example].T

        sampler.n_samples = data.shape[0] * sampling_factor
        sampler.n_samples *= model_class().min_sample_size
        ransac_gen = sampling.ModelGenerator(model_class, data, sampler)
        ac_tester = tester_class(data, epsilon, inliers_threshold)

        match = re.match(ex_type + '[0-9]*_', example)
        try:
            match = re.match('[0-9]+', match.group()[len(ex_type):])
            n_groups = int(match.group())
        except AttributeError:
            n_groups = 4
        gt_groups = ground_truth(model_class, data, inliers_threshold, n_groups)

        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print('seed:', seed)
        np.random.seed(seed)

        res = test(model_class, data, example, ransac_gen, ac_tester, gt_groups)
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


if __name__ == '__main__':
    run_all(oversampling=1)
    plt.show()