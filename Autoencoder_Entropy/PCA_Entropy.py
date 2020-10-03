from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as pl
import numpy as np
import csv
import math
import sys
from sklearn.decomposition import PCA

def get_real_values(file):

    values = np.array([[]])
	
    with open(file) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for idx,row in enumerate(csv_reader):
            if (idx == 0):
                values = np.array([row],dtype="f")
                continue
            if row == []:
                continue
            rw = np.array([row],dtype="f")
            values = np.concatenate((values,rw))

    return values
	
	
if True:

	file = 'text_30_9_1.csv'
	kde = KernelDensity(kernel='gaussian',bandwidth=0.37)

	values = get_real_values(file)
	pca = PCA(whiten=True,n_components=5)
	
	print(values.size)
	
	
	for idx in [50,100,150,200]:
	
		entr_arr = values[0:(idx*100)]
			
		pca.fit(entr_arr)
		kde.fit(pca.components_)
		
		b = pca.components_
		
		print(b)
		
		probabilites = kde.score_samples(pca.components_)
		probabilites = np.exp(probabilites)
		
		print(probabilites)
	
		total_entropy = 0

		for k in probabilites:
			total_entropy += -k * math.log(2, k)
				
		print(total_entropy)

	
	
	
	
	
	
	
	
	
	
	