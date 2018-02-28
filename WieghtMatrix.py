import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances


def construct_S(X, **kwargs):
	
	n_samples, n_features = np.shape(X)
	
	k=kwargs['neighbour_size']
	
	t = kwargs['t_param']
     
    # compute pairwise euclidean distances
	D = pairwise_distances(X)
	#print X
	D **= 2
	
	
	# sort the distance matrix D in ascending order
	Dsorted = np.sort(D, axis=1)
	idxSorted = np.argsort(D, axis=1)
	idx = idxSorted[:, 0:k+1]# returns the index of entire matrix
	dist = Dsorted[:, 0:k+1]
	
	# compute the pairwise heat kernel distances
	Weight = np.exp(-dist/(t))
	G = np.zeros((n_samples*(k+1),3))
	G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
	G[:, 1] = np.ravel(idx, order='F')
	G[:, 2] = np.ravel(Weight, order='F')
	
	# build the sparse affinity matrix W
	S = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
	bigger = np.transpose(S) > S
	S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
	print S
	return S
