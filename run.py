import argparse
import utils

parser = argparse.ArgumentParser(description='Quantized Nearest Neighbor')

parser.add_argument('--dataset', type=str, default="iris", help='Dataset for Nearest Neighbor')
parser.add_argument('--d_metric', type=str, default="cosine", help='Distance Metric to use Nearest Neighbor')
parser.add_argument('--qbits', type=int, default=0, help='Number of bits for quantization')

args = parser.parse_args()

if args.d_metric == "mcam":
    if args.qbits != 4 and args.qbits != 3:
        raise Exception("MCAM only supports quantization bits up to 4")

print("-----------------------------------------------")
print("Distance Metric: ", args.d_metric)
print("Dataset: ", args.dataset)
print("Quantization", args.qbits)
print("-----------------------------------------------")

if args.d_metric == "euclidean":
    minkowski_p = 2
if args.d_metric == "manhattan":
    minkowski_p = 1

qbits = args.qbits
utils.run_knn(args)





#dataset = sys.argv[1]
#d_metric = sys.argv[2]
#if len(sys.argv) > 3:
#    qbits = int(sys.argv[3])
#else:
#    qbits = 0
#parser.add_argument('', action="store_true", default=False)
#parser.add_argument('-b', action="store", dest="b")
#parser.add_argument('-c', action="store", dest="c", type=int)

#print parser.parse_args(['-a', '-bval', '-c', '3'])
