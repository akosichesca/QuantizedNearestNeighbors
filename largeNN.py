import numpy as np
import h5py
import distance_metric as dm
import sys
from sklearn.neighbors import KNeighborsClassifier
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Quantized Nearest Neighbor')

parser.add_argument('--dataset', type=str, default="mnist", help='Dataset for Nearest Neighbor')
parser.add_argument('--d_metric', type=str, default="cosine", help='Distance Metric to use Nearest Neighbor')
parser.add_argument('--qbits', type=int, default=0, help='Number of bits for quantization')

args = parser.parse_args()

if args.d_metric == "mcam":
    if args.qbits != 4 and args.qbits != 3:
        raise Exception("MCAM only supports quantization bits up to 4")

qbits = args.qbits
dataset = args.dataset
d_metric = args.d_metric

print("-----------------------------------------------")
print("Distance Metric: ", args.d_metric)
print("Dataset: ", args.dataset)
print("Quantization", args.qbits)
print("-----------------------------------------------")


if dataset == 'mnist':
    hf = h5py.File('mnist-784-euclidean.hdf5', 'r')
elif dataset == 'glove':
    hf = h5py.File('glove-25-angular.hdf5', 'r')
elif dataset == 'sift':
    hf = h5py.File('sift-128-euclidean.hdf5', 'r')


nneighbor = np.array(hf.get('neighbors'))
distances = np.array(hf.get('distances'))
X_train = np.array(hf.get('train'))
X_test = np.array(hf.get('test'))


k = 100
ntrain = len(X_train)
ntest = 10 #len(X_test)
d = [0] * ntrain
hits = 0
for i in tqdm.tqdm(range(ntest)):
    for j in range(ntrain):
        d[j] = dm.minkowski(X_test[i], X_train[j])
    topk = np.argsort(d)[:100]
    hits = hits + len(np.intersect1d(topk, nneighbor[i]))

print("Accuracy: ", hits/ntest)
