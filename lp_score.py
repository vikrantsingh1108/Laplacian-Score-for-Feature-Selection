# -*- coding: utf-8 -*-
import numpy

from scipy.sparse import *
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph as kng
from scipy.sparse.linalg import expm
from scipy.linalg import solve_banded
from scipy.spatial.distance import pdist
import scipy.spatial.distance
import math


def construct_W(X, **kwargs):
	n_samples, n_features = numpy.shape(X)
	k=kwargs['neighbour_size']
	t = kwargs['t_param']
	S=kng(X, k+1, mode='distance',metric='euclidean') #sqecludian distance works only with mode=connectivity  results were absurd
	S = (-1*(S*S))/(2*t*t)
	S=S.tocsc()
	S=expm(S)
	S=S.tocsr()
	
	
	#[1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
	#Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method
	 
	bigger = numpy.transpose(S) > S
	S = S - S.multiply(bigger) + numpy.transpose(S).multiply(bigger)
	return S

	
def lap_score(X, **kwargs):
	
	
    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
		
		if 't_param' not in kwargs.keys():
			t_param=1
		else:
			t = kwargs['t_param']
		
		if 'neighbour_size' not in kwargs.keys():
			neighbour_size=5
		else:
			n=kwargs['neighbour_size']
			
		W = construct_W(X,t_param=t,neighbour_size=n)
		
    # construct the affinity matrix W
    else:
		W = kwargs['W']
    
    # build the diagonal D matrix from affinity matrix W
    D = numpy.array(W.sum(axis=1))
    
    
    L = W
    
    
    tmp = numpy.dot(numpy.transpose(D),X)
    D = diags(numpy.transpose(D), [0])
    Xt = numpy.transpose(X)
    t1 = numpy.transpose(numpy.dot(Xt, D.todense()))
    t2 = numpy.transpose(numpy.dot(Xt, L.todense()))
    # compute the numerator of Lr
    tmp=numpy.multiply(tmp, tmp)/D.sum()
    D_prime = numpy.sum(numpy.multiply(t1, X), 0) -tmp 
    # compute the denominator of Lr
    L_prime = numpy.sum(numpy.multiply(t2, X), 0) -tmp 
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    # compute laplacian score for all features
    score = 1 - numpy.array(numpy.multiply(L_prime, 1/D_prime))[0, :]
    return numpy.transpose(score)

"""
    Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
    important the feature is
"""
def feature_ranking(score):
    idx = numpy.argsort(score, 0)
    return idx+1

def LaplacianScore(X, **kwargs):
	
	if 'W' not in kwargs.keys():
		
		if 't_param' not in kwargs.keys():
			t_param=1
		else:
			t = kwargs['t_param']
		
		if 'neighbour_size' not in kwargs.keys():
			neighbour_size=5
		else:
			n=kwargs['neighbour_size']
			
		W = construct_W(X,t_param=t,neighbour_size=n)
		n_samples, n_features = numpy.shape(X)
    # construct the affinity matrix W
	else:
		W = kwargs['W']
    
    #construct the diagonal matrix
	D=numpy.array(W.sum(axis=1))
	D = diags(numpy.transpose(D), [0])
	#construct graph Laplacian L
	L=D-W.toarray()
	
	#construct 1= [1,···,1]' 
	I=numpy.ones((n_samples,n_features))
	
	#construct fr' => fr= [fr1,...,frn]'
	Xt = numpy.transpose(X)
	
	#construct fr^=fr-(frt D I/It D I)I
	t=numpy.matmul(numpy.matmul(Xt,D.toarray()),I)/numpy.matmul(numpy.matmul(numpy.transpose(I),D.toarray()),I)
	t=t[:,0]
	t=numpy.tile(t,(n_samples,1))
	fr=X-t
	
	#Compute Laplacian Score
	fr_t=numpy.transpose(fr)
	Lr=numpy.matmul(numpy.matmul(fr_t,L),fr)/numpy.matmul(numpy.dot(fr_t,D.toarray()),fr)
	
	return numpy.diag(Lr)
	
