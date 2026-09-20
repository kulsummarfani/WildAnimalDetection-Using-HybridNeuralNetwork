"""Group the augmented images in Animals/ by the source photo they came from.

Augmentation.py made ~10 augmented copies of every source photo, and the
original photos were not kept. A random train/test split therefore puts
near-copies of the same photo on both sides (data leakage). This script
recovers the source groups so the split can be done per group instead.

How: ImageDataGenerator keeps image dimensions, so siblings share exact
(height, width). Small same-size sets are one source; larger ones are split with
SIFT feature matching (flip-aware, RANSAC). Errors here only ever over-merge
sources (safe); a wrongly split source would be the harmful case.

Outputs (model/): grouped_X.npy (N,32,32,3 uint8 BGR), grouped_Y.npy,
grouped_G.npy (group id per image) and grouped_files.csv (audit trail).

Run once:  .venv/bin/python make_groups.py
"""

import collections
import csv
import itertools
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Animals'
OUT_DIR = ROOT / 'model'
LABELS = ('Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger')
IMG_SIZE = 32
MAX_SINGLE_SOURCE = 12
MIN_INLIERS = 15


def _features(path):
    """SIFT keypoints/descriptors for the image and its horizontal flip."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    scale = 256 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
    sift = cv2.SIFT_create(nfeatures=400)
    out = []
    for image in (gray, cv2.flip(gray, 1)):
        keypoints, descriptors = sift.detectAndCompute(image, None)
        out.append((np.float32([k.pt for k in keypoints]), descriptors))
    return out


def _inliers(a, b, matcher):
    """Best RANSAC inlier count between image a and image b (or its flip)."""
    best = 0
    points_a, desc_a = a[0]
    for points_b, desc_b in b:
        if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
            continue
        pairs = [p for p in matcher.knnMatch(desc_a, desc_b, k=2) if len(p) == 2]
        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        src = points_a[[m.queryIdx for m in good]]
        dst = points_b[[m.trainIdx for m in good]]
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best


def _split_same_size(files):
    """Cluster one same-size set of files into sources. Returns a local id per file."""
    if len(files) <= MAX_SINGLE_SOURCE:
        return [0] * len(files)
    cv2.setNumThreads(1)
    feats = [_features(f) for f in files]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    adjacency = lil_matrix((len(files), len(files)))
    for i, j in itertools.combinations(range(len(files)), 2):
        if _inliers(feats[i], feats[j], matcher) >= MIN_INLIERS:
            adjacency[i, j] = 1
    return connected_components(adjacency, directed=False)[1].tolist()


def _job(job):
    label, shape, files = job
    return label, shape, files, _split_same_size(files)


def main():
    jobs = []
    for label in LABELS:
        by_size = collections.defaultdict(list)
        for path in sorted((DATA_DIR / label).glob('*.png')):
            image = cv2.imread(str(path))
            if image is None:
                print(f'skipping unreadable file: {path}')
                continue
            by_size[image.shape[:2]].append(path)
        jobs += [(label, shape, files) for shape, files in sorted(by_size.items())]
    jobs.sort(key=lambda job: -len(job[2]))
    print(f'{sum(len(j[2]) for j in jobs)} images, {len(jobs)} same-size sets')

    with Pool() as pool:
        results = pool.map(_job, jobs, chunksize=1)

    rows, next_group = [], 0
    for label, shape, files, local in sorted(results, key=lambda r: (r[0], r[1])):
        remap = {}
        for path, local_id in zip(files, local):
            if local_id not in remap:
                remap[local_id] = next_group
                next_group += 1
            rows.append((path, LABELS.index(label), remap[local_id]))
    rows.sort(key=lambda row: str(row[0]))

    X = np.stack([cv2.resize(cv2.imread(str(p)), (IMG_SIZE, IMG_SIZE)) for p, _, _ in rows])
    Y = np.array([y for _, y, _ in rows])
    G = np.array([g for _, _, g in rows])
    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / 'grouped_X.npy', X)
    np.save(OUT_DIR / 'grouped_Y.npy', Y)
    np.save(OUT_DIR / 'grouped_G.npy', G)
    with open(OUT_DIR / 'grouped_files.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file', 'label', 'group'])
        writer.writerows((p.relative_to(ROOT).as_posix(), LABELS[y], g) for p, y, g in rows)

    sizes = np.bincount(G)
    print(f'{len(sizes)} source groups | size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}')
    print('group sizes:', dict(sorted(collections.Counter(sizes.tolist()).items())))
    mixed = [g for g in range(len(sizes)) if len(set(Y[G == g])) > 1]
    print('groups spanning more than one class (should be 0):', len(mixed))
    oversized = int((sizes > 2 * MAX_SINGLE_SOURCE).sum())
    if oversized:
        print(f'note: {oversized} groups are larger than {2 * MAX_SINGLE_SOURCE} (over-merged; safe for the split)')


if __name__ == '__main__':
    main()
