import torch
import numpy as np
import tqdm

class LSH:
    def __init__(self,size,qbits=32):
        torch.manual_seed(111)
        self.dim = size
        self.lsh_nHashes = 6 #np.ceil(np.log2(size))
        self.lsh_nBuckets = 2**self.lsh_nHashes
        self.lsh_multiplier = torch.from_numpy(np.asarray(self.binlist2int(2,list(range(self.lsh_nHashes-1,-1,-1)))))
        self.lsh_nLibraries = qbits
        print(self.lsh_nLibraries)
        self.random_projections = [torch.FloatTensor(self.dim, self.lsh_nHashes).normal_(mean=0, std=2) for i in range(self.lsh_nLibraries)]
        self.lsh_bucket_loc = []
        self.key = []
    def signature(self,x):
        # note that this is in long for faster processing
        return [torch.le(torch.matmul(x,self.random_projections[l]),0).long() for l in range(self.lsh_nLibraries)]
    def binlist2int(self,exp,mylist):
        return [ exp**x for x in mylist ]
    def append(self,x,y):
        signature = self.signature(x)
        self.lsh_bucket_loc.append(torch.stack([torch.matmul(signature[l],self.lsh_multiplier) for l in range(self.lsh_nLibraries)]))
        self.key.append(y)
    def search(self,x):
        n = len(self.lsh_bucket_loc)        
        signature = self.signature(x) 
        sig_locs = torch.stack([torch.matmul(signature[l],self.lsh_multiplier) for l in range(self.lsh_nLibraries)])
        dist = torch.zeros([n,1])         
        for j in range(n):
            sig_train = self.lsh_bucket_loc[j]
            dist[j] = (sig_locs-sig_train == 0).sum()
        return self.key[torch.argmax(dist)]
    def fit(self,X,Y):
        for i in tqdm.tqdm(range(len(X))):
            self.append(torch.from_numpy(X[i]).float(),Y[i])
        return 0
    def predict(self,X):
        y = [0]*len(X)
        for i in tqdm.tqdm(range(len(X))):
            y[i] = self.search(torch.from_numpy(X[i]).float())
        return y
    
