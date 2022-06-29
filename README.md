# QuantizedNearestNeighbors

This is a code to implement a simple Quantized Nearest Neighbor Search. 

There are three commmand line arguments: 

-d-metric : which determines what distance function to use

-qbits : number of bits to represent each dimension

-dataset: name of dataset to use

python run.py -d-metric [string] -qbits [number] -dataset [number]

(all commandline arguments can be removed)

-d-metric [string]
[string] : dot, cosine, euclidean, manhattan, mcam, mcam_ideal, lsh
If not included uses dot product.

-qbits [number]
[num] number of quantized bits
It will do full precision if not included as argument.

-dataset [number]
[string] : cancer, dermatology, glass, ionosphere, iris, liver, soybean, adult, heart, wine
If not included uses wine dataset.
