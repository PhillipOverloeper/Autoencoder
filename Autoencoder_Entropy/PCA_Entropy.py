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

    file = 'state_action_pca_2.csv'
    kde = KernelDensity(kernel='gaussian',bandwidth=0.5)

    values = get_real_values(file)
    pca = PCA(whiten=True,n_components=5)
    pca.fit(values)
    
    for idx in [50,100,150,200]:
    
        entr_arr = values[0:(idx*100)]
        a = np.array([[]])
        b = np.array([[]])
        c = np.array([[]])
        d = np.array([[]])
        e = np.array([[]])
            
        vector = pca.transform(entr_arr)

        for j in range(len(vector)):
			
            a = np.append(a,vector[j][0])
            b = np.append(b,vector[j][1])
            c = np.append(c,vector[j][2])
            d = np.append(d,vector[j][3])
            e = np.append(e,vector[j][4])
            
            
        for i in [a,b,c,d,e]:
            
            t = np.arange(-3,3,1/1000)
            
            for idy,k in enumerate(i):
                for l in t:
                    if k < l:
                        i[idy] = l
                        break
        
            i = np.asarray(i)
            i = i.reshape((len(i), 1))        
        
        
		
            kde.fit(i)#
		
            probabilites = kde.score_samples(i)
            probabilites = np.exp(probabilites)#
		
 #           print(probabilites)#
	
            total_entropy = 0

            for k in probabilites:
                if k == 0:
                    continue
                total_entropy += -k * math.log(2, k)
				
            print('This is the entropy: ' + str(total_entropy) + 'of the ' + str(idx))
            
            
            with open('pca_entropy_pca2.csv',mode='a') as file:
                writer = csv.writer(file,delimiter=",")
                writer.writerow(str(total_entropy))

	
	
	
	
	
	
	
	
	
	
	