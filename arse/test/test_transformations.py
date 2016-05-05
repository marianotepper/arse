from __future__ import absolute_import, print_function
import os
import PIL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn.apionly as sns
import numpy as np
import scipy.io
import pickle
import timeit
import arse.biclustering as bc
import arse.test.utils as test_utils
import arse.pme.preference as pref


def base_plot(data):
    def inner_plot_img(pos, img):
        pos_rc = pos + np.array(img.shape[:2], dtype=np.float) / 2
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.hold(True)
        plt.imshow(gray_image, cmap='gray')
        plt.scatter(pos_rc[:, 0], pos_rc[:, 1], c='w', marker='o', s=10)
        plt.axis('off')

    x = data['data']
    plt.figure()
    plt.subplot(121)
    inner_plot_img(x[:, 0:2], data['img1'])
    plt.subplot(122)
    inner_plot_img(x[:, 3:5], data['img2'])


def plot_models(data, groups, palette, s=10, marker='o'):
    def inner_plot_img(pos, img):
        pos_rc = pos + np.array(img.shape[:2], dtype=np.float) / 2
        plt.hold(True)
        gray_image = PIL.Image.fromarray(img).convert('L')
        plt.imshow(gray_image, cmap='gray', interpolation='none')
        for g, color in zip(groups, palette):
            plt.scatter(pos_rc[g, 0], pos_rc[g, 1], c=color, edgecolors='face',
                        marker=marker, s=s)
        plt.axis('off')

    x = data['data']
    plt.figure()
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0)
    plt.subplot(gs[0])
    inner_plot_img(x[:, 0:2], data['img1'])
    plt.subplot(gs[1])
    inner_plot_img(x[:, 3:5], data['img2'])


def line_plot(data, groups, palette, s=2, marker='o'):
    x = data['data']
    x1 = x[:, :2] + np.array(data['img1'].shape[:2], dtype=np.float) / 2
    x2 = x[:, 3:5] + np.array(data['img2'].shape[:2], dtype=np.float) / 2
    img = np.hstack((data['img1'], data['img2']))
    width1 = data['img1'].shape[1]

    plt.figure()
    plt.hold(True)
    gray_image = PIL.Image.fromarray(img).convert('L')
    plt.imshow(gray_image, cmap='gray', interpolation='none')
    for g, color in zip(groups, palette):
        for i in np.where(g)[0]:
            plt.plot([x1[i, 0], x2[i, 0] + width1], [x1[i, 1], x2[i, 1]],
                     c=color, linewidth=.5, marker=marker, markersize=s,
                     markerfacecolor='none', markeredgecolor=color, alpha=.5)
    plt.axis('off')


def ground_truth(labels):
    gt_groups = []
    for i in np.unique(labels):
        gt_groups.append(labels == i)
    return gt_groups


def run_biclustering(model_class, data, pref_matrix, comp_level, thresholder,
                     ac_tester, output_prefix, palette='Set1'):
    t = timeit.default_timer()
    bic_list = bc.bicluster(pref_matrix, comp_level=comp_level)
    t1 = timeit.default_timer() - t
    print('Time:', t1)

    bic_list = test_utils.clean(model_class, data['data'], thresholder,
                                ac_tester, bic_list, share_elements=False)[1]

    colors = sns.color_palette(palette, len(bic_list))

    plt.figure()
    pref.plot(pref_matrix, bic_list=bic_list, palette=colors)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    bc_groups = [np.squeeze(bic[0].toarray()) for bic in bic_list]

    plot_models(data, bc_groups, palette=colors)
    plt.savefig(output_prefix + '_final_models.pdf', dpi=600)

    line_plot(data, bc_groups, palette=colors)
    plt.savefig(output_prefix + '_line_plot.pdf', dpi=600)

    if bc_groups:
        inliers = reduce(lambda a, b: np.logical_or(a, b), bc_groups)
    else:
        inliers = np.zeros((pref_matrix.shape[0],), dtype=np.bool)
    if 'label' in data:
        bc_groups.append(np.logical_not(inliers))
        gt_groups = ground_truth(data['label'])
        gnmi, prec, rec = test_utils.compute_measures(gt_groups, bc_groups)
        return dict(time=t1, gnmi=gnmi, precision=prec, recall=rec)
    else:
        return dict(time=t1)


def test(model_class, data, name, ransac_gen, thresholder, ac_tester,
         dir_name):
    x = data['data']
    print(name, x.shape)

    output_dir = '../results/{0}/'.format(dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_prefix = output_dir + name

    base_plot(data)
    plt.savefig(output_prefix + '_data.pdf', dpi=600)

    if 'label' in data:
        gt_groups = ground_truth(data['label'])
        gt_colors = sns.color_palette('Set1', len(gt_groups) - 1)
        gt_colors.insert(0, [1., 1., 1.])
        plot_models(data, gt_groups, palette=gt_colors)
        plt.savefig(output_prefix + '_gt10.pdf', dpi=600)
        plot_models(data, gt_groups, palette=gt_colors, s=.1, marker='.')
        plt.savefig(output_prefix + '_gt1.pdf', dpi=600)

    pref_matrix, _ = pref.build_preference_matrix(ransac_gen, thresholder,
                                                  ac_tester)

    scipy.io.savemat(output_prefix + '.mat', {'pref_matrix': pref_matrix})
    with open(output_prefix + '.pickle', 'wb') as handle:
        pickle.dump(pref_matrix, handle)
    with open(output_prefix + '.pickle', 'rb') as handle:
        pref_matrix = pickle.load(handle)

    print('Preference matrix size:', pref_matrix.shape)

    plt.figure()
    pref.plot(pref_matrix)
    plt.savefig(output_prefix + '_pref_mat.pdf', dpi=600)

    print('Running regular bi-clustering')
    compression_level = None
    stats_reg = run_biclustering(model_class, data, pref_matrix,
                                 compression_level, thresholder, ac_tester,
                                 output_prefix + '_bic_reg')

    print('Running compressed bi-clustering')
    compression_level = 32
    stats_comp = run_biclustering(model_class, data, pref_matrix,
                                  compression_level, thresholder, ac_tester,
                                  output_prefix + '_bic_comp')

    return stats_reg, stats_comp
