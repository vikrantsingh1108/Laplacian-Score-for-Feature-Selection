from lp_score import *
import argparse as aP
import numpy as np
import time

def ArgumenParser():
	parser = aP.ArgumentParser(description='Parse the Command line arguments')
	parser.add_argument('-f', action="store", dest="DataSet" , default='IRIS.csv')
	parser.add_argument('-k', action="store", dest="neighbour_size",type=int,default=16)
	parser.add_argument('-t', action="store", dest="t_param" , type =int,default=2)
	return parser.parse_args()
	
if __name__=='__main__':
	start_time=time.time()
	param=ArgumenParser()
	X=np.loadtxt(param.DataSet,delimiter=',')
	n_samples,n_feature=X.shape
	data=X[:,0:n_feature-1]
	#Y=lap_score(data,neighbour_size=param.neighbour_size,t_param=param.t_param)
	#Z =  feature_ranking(Y)
	#print Y
	#print "\n"
	#print Z
	L=LaplacianScore(data,neighbour_size=param.neighbour_size,t_param=param.t_param)
	print L
	print feature_ranking(L)
	print("--- %s seconds ---" % (time.time() - start_time))

