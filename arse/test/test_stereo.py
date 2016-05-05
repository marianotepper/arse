import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import arse.pme.sampling as sampling
import arse.pme.multigs as multigs
import arse.pme.membership as membership
import arse.pme.fundamental as fundamental
import arse.pme.acontrario as ac
import arse.test.test_transformations as test_transformations
import arse.test.test_utils as test_utils


def keypoints_and_descriptors(img):
    detector = cv2.BRISK_create()
    kp = detector.detect(img, None)
    return detector.compute(img, kp)


def to_array(data_fun, list):
    return np.array(map(data_fun, list))


def trim_matches(match_list, tol=0.6):
    return [ms[0] for ms in match_list if ms[0].distance < tol * ms[1].distance]


def load(path):
    img1 = cv2.imread(path + 'im0.png', 0)
    img2 = cv2.imread(path + 'im1.png', 0)

    kp1, desc1 = keypoints_and_descriptors(img1)
    kp2, desc2 = keypoints_and_descriptors(img2)

    flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    img_plot = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=flag)
    cv2.imwrite(path + '_kp1.png', img_plot)
    img_plot = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=flag)
    cv2.imwrite(path + '_kp2.png', img_plot)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    img_plot = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                               matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0))
    cv2.imwrite(path + '_matches.png', img_plot)

    pts1 = to_array(lambda e: e.pt, kp1)
    pts2 = to_array(lambda e: e.pt, kp2)
    match_arr = to_array(lambda e: (e.queryIdx, e.trainIdx), matches)

    pts1 = pts1[match_arr[:, 0], :] - np.array(img1.shape, dtype=np.float) / 2
    pts2 = pts2[match_arr[:, 1], :] - np.array(img2.shape, dtype=np.float) / 2
    all_ones = np.ones(shape=(match_arr.shape[0], 1))
    x = np.hstack((pts1, all_ones, pts2, all_ones))

    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
    dists = []
    for i in range(match_arr.shape[0]):
        epi_lines1 = np.dot(F, x[i, :3])
        epi_lines1 /= np.linalg.norm(epi_lines1[:2])
        dists.append(np.abs(np.sum(x[i, 3:] * epi_lines1)))

    plt.figure()
    plt.plot(dists)

    data = dict(data=x, img1=img1, img2=img2)
    return data


def run(name):
    logger = test_utils.Logger('test_stereo.txt')
    sys.stdout = logger

    inliers_threshold = 50.
    n_samples = 5000
    epsilon = 0

    print(name)

    path = '../data/{0}/'.format(name)
    data = load(path)

    model_class = fundamental.Fundamental
    img_size = data['img2'].shape[:2]
    nfa_proba = (2. * np.linalg.norm(img_size) / np.prod(img_size))

    sampler = sampling.UniformSampler(n_samples)
    generator = sampling.ModelGenerator(model_class, data['data'], sampler)
    # generator = multigs.ModelGenerator(model_class, data['data'], n_samples)
    min_sample_size = model_class().min_sample_size
    ac_tester = ac.ImageTransformNFA(epsilon, nfa_proba, min_sample_size)
    thresholder = membership.GlobalThresholder(inliers_threshold)

    seed = 0
    # seed = np.random.randint(0, np.iinfo(np.uint32).max)
    print('seed:', seed)
    np.random.seed(seed)

    prefix = name
    test_transformations.test(model_class, data, prefix, generator, thresholder,
                              ac_tester, name)

    plt.close('all')
    sys.stdout = logger.stdout
    logger.close()


if __name__ == '__main__':
    run('Flowers-perfect')
    run('Mask-perfect')
    plt.show()