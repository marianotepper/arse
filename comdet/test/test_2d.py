from __future__ import absolute_import
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import scipy.sparse as sp
import numpy as np
import scipy.io
import re
import timeit
import comdet.biclustering.preference as pref
import comdet.biclustering.nmf as bc
import comdet.biclustering.deflation as deflation
import comdet.test.utils as test_utils
import comdet.pme.line as line
import comdet.pme.circle as circle
import comdet.pme.sampling as sampling
import comdet.pme.acontrario.line as ac_line
import comdet.pme.acontrario.circle as ac_circle


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


def ground_truth(n_elements, n_groups=5, group_size=50):
    gt_groups = []
    for i in range(n_groups):
        v = np.zeros((n_elements,), dtype=bool)
        v[i * group_size:(i+1) * group_size] = True
        gt_groups.append(v)
    return gt_groups


def run_biclustering(model_class, x, original_models, pref_matrix, deflator,
                     ac_tester, gt_groups, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(deflator)
    t1 = timeit.default_timer() - t
    print 'Time:', t1

    mod_inliers_list, bic_list = test_utils.clean(model_class, x, ac_tester,
                                                  bic_list)

    palette = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot_preference_matrix(pref_matrix, bic_list=bic_list, palette=palette)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    plot_final_models(x, [mi[0] for mi in mod_inliers_list], palette=palette)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    plot_original_models(x, original_models, [bic[1] for bic in bic_list],
                         palette)
    plt.savefig(output_prefix + '_bundles.pdf', dpi=600)

    test_utils.compute_measures(gt_groups, [bic[0] for bic in bic_list])


def test(model_class, x, name, ransac_gen, ac_tester, gt_groups):
    print name, x.shape

    output_prefix = '../results/' + name

    base_plot(x)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    pref_matrix, orig_models = test_utils.build_preference_matrix(x.shape[0],
                                                                  ransac_gen,
                                                                  ac_tester)
    print 'Preference matrix size:', pref_matrix.shape

    base_plot(x)
    plot_models(orig_models, alpha=0.2)
    plt.savefig(output_prefix + '_original_models.pdf', dpi=600)

    plt.figure()
    pref.plot_preference_matrix(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print 'Running regular bi-clustering'
    deflator = deflation.Deflator(pref_matrix)
    run_biclustering(model_class, x, orig_models, pref_matrix, deflator,
                     ac_tester, gt_groups, output_prefix + '_bic_reg')

    print 'Running compressed bi-clustering'
    compression_level = 128
    deflator = deflation.L1CompressedDeflator(pref_matrix, compression_level)
    run_biclustering(model_class, x, orig_models, pref_matrix, deflator,
                     ac_tester, gt_groups, output_prefix + '_bic_comp')


if __name__ == '__main__':
    # plt.switch_backend('TkAgg')
    sampling_factor = 20
    inliers_threshold = 0.03
    epsilon = 0

    configuration = {'Star': (line.Line, sampling.UniformSampler(),
                              ac_line.LocalNFA),
                     'Stairs': (line.Line, sampling.UniformSampler(),
                                ac_line.LocalNFA),
                     'Circles': (circle.Circle, sampling.UniformSampler(),
                                 ac_circle.LocalNFA),
                     }

    # configuration = {'Star': (line.Line, sampling.UniformSampler(),
    #                           ac_line.LocalNFA),
    #                  'Stairs': (line.Line,
    #                             sampling.GaussianLocalSampler(0.05),
    #                             ac_line.LocalNFA),
    #                  'Circles': (circle.Circle,
    #                              sampling.GaussianLocalSampler(0.5),
    #                              ac_circle.LocalNFA),
    #                  }
    #
    # configuration = {'Star': (line.Line, sampling.UniformSampler(),
    #                           ac_line.GlobalNFA),
    #                  'Stairs': (line.Line,
    #                             sampling.GaussianLocalSampler(0.05),
    #                             ac_line.GlobalNFA),
    #                  'Circles': (circle.Circle,
    #                              sampling.GaussianLocalSampler(0.5),
    #                              ac_circle.GlobalNFA),
    #                  }

    examples = scipy.io.loadmat('../data/JLinkageExamples.mat')
    for current_example in examples:
        exp_type = None
        for c in configuration:
            if current_example.find(c) == 0:
                exp_type = c
                break
        else:
            if exp_type is None:
                continue

        model_class, sampler, ac_tester_class = configuration[exp_type]
        data = examples[current_example].T

        sampler.n_samples = data.shape[0] * sampling_factor
        ransac_gen = sampling.ransac_generator(model_class, data, sampler,
                                               inliers_threshold)
        ac_tester = ac_tester_class(data, epsilon, inliers_threshold)

        match = re.match(exp_type + '[0-9]*_', current_example)
        try:
            match = re.match('[0-9]+', match.group()[len(exp_type):])
            n_groups = int(match.group())
        except AttributeError:
            n_groups = 4
            continue
        gt_groups = ground_truth(data.shape[0], n_groups=n_groups,
                                 group_size=50)

        print '-'*40
        seed = 0
        # seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print 'seed:', seed
        np.random.seed(seed)

        test(model_class, data, current_example, ransac_gen, ac_tester,
             gt_groups)

    # plt.show()