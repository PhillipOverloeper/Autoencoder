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

    file_1 = 'state_action_pca_2.csv'
    file_2 = 'state_action_autoencoder_3.csv' 
    file_3 = 'state_action_baseline_3.csv'

    values_1 = get_real_values(file_1)
    values_2 = get_real_values(file_2)
    values_3 = get_real_values(file_3)
    pca = PCA(whiten=True,n_components=5)
    pca.fit(values_1)
    
    for episode in [50,100,150,200]:
        
        # Get date until a certain episode
        entr_arr_1 = values_1[0:(episode*100)]
        entr_arr_2 = values_2[0:(episode*100)]
        entr_arr_3 = values_3[0:(episode*100)]
        # Initialize the vectors for the dimensions
        a_1 = np.array([[]])
        b_1 = np.array([[]])
        c_1 = np.array([[]])
        d_1 = np.array([[]])
        e_1 = np.array([[]])
        a_2 = np.array([[]])
        b_2 = np.array([[]])
        c_2 = np.array([[]])
        d_2 = np.array([[]])
        e_2 = np.array([[]])
        a_3 = np.array([[]])
        b_3 = np.array([[]])
        c_3 = np.array([[]])
        d_3 = np.array([[]])
        e_3 = np.array([[]])
        # Perform dimensionality reduction 
        vector_1 = pca.transform(entr_arr_1)
        vector_2 = pca.transform(entr_arr_2)
        vector_3 = pca.transform(entr_arr_3)
        # Fill the different dimensions
        for i in range(len(vector_1)):
			
            # Dimensions for PCA
            a_1 = np.append(a_1,vector_1[i][0])
            b_1 = np.append(b_1,vector_1[i][1])
            c_1 = np.append(c_1,vector_1[i][2])
            d_1 = np.append(d_1,vector_1[i][3])
            e_1 = np.append(e_1,vector_1[i][4])
            # Dimensions for Autoencoder
            a_2 = np.append(a_2,vector_2[i][0])
            b_2 = np.append(b_2,vector_2[i][1])
            c_2 = np.append(c_2,vector_2[i][2])
            d_2 = np.append(d_2,vector_2[i][3])
            e_2 = np.append(e_2,vector_2[i][4])
            # Dimensions for Baseline
            a_3 = np.append(a_3,vector_3[i][0])
            b_3 = np.append(b_3,vector_3[i][1])
            c_3 = np.append(c_3,vector_3[i][2])
            d_3 = np.append(d_3,vector_3[i][3])
            e_3 = np.append(e_3,vector_3[i][4])  
            
        # Get min and max value
        min_value_1 = min([min(a_1),min(a_2),min(a_3)])
        print(min(a_1))
        print(min(a_2))
        print(min(a_3))
        print(min_value_1)
        min_value_2 = min([min(b_1),min(b_2),min(b_3)])
        min_value_3 = min([min(c_1),min(c_2),min(c_3)])
        min_value_4 = min([min(d_1),min(d_2),min(d_3)])
        min_value_5 = min([min(e_1),min(e_2),min(e_3)])
        max_value_1 = max([max(a_1),max(a_2),max(a_3)])
        max_value_2 = max([max(b_1),max(b_2),max(b_3)])
        max_value_3 = max([max(c_1),max(c_2),max(c_3)])
        max_value_4 = max([max(d_1),max(d_2),max(d_3)])
        max_value_5 = max([max(e_1),max(e_2),max(e_3)])
            
        for iteration in range(3):
        
            if iteration == 0:
                a = a_1
                b = b_1
                c = c_1
                d = d_1
                e = e_1
                
                print('')
                print('PCA of ' + str(episode))
                print('')
            elif iteration == 1:
                a = a_2
                b = b_2
                c = c_2
                d = d_2
                e = e_2
     
                print('')     
                print('Autoencoder of ' + str(episode))
                print('')         
            else:
                a = a_3
                b = b_3
                c = c_3
                d = d_3
                e = e_3  

                print('')
                print('Baseline of ' + str(episode))
                print('')                
            
            for n_dim,dim in enumerate([a,b,c,d,e]):
            
                if n_dim == 0:
                    min_value = min_value_1
                    max_value = max_value_1
                elif n_dim == 1:
                    min_value = min_value_2
                    max_value = max_value_2
                elif n_dim == 2:
                    min_value = min_value_3
                    max_value = max_value_3
                elif n_dim == 3:
                    min_value = min_value_4
                    max_value = max_value_4
                else:
                    min_value = min_value_5
                    max_value = max_value_5
                   
            
                disc_values = np.arange(min_value,max_value,(max_value - min_value)/1000)
                numb_values = np.zeros(len(disc_values))            
                
                for idx,i in enumerate(dim):
                    for idj,j in enumerate(disc_values):
                        if i <= j:
                            numb_values[idj] += 1
                            break
              
                
                probi = np.array([])

                total_entropy = 0
                tot_prob = 0

                for zahl,prob in enumerate(numb_values):
                    p = prob/(episode*100)
                    probi = np.append(probi,p)
                    if p == 0:   
                        continue
                    tot_prob += p
                    total_entropy += -p * math.log(p, 2)
                    
                print('This is the entropy ' + str(total_entropy) + ' of the ' + str(n_dim+1))
                print(np.mean(probi,axis=0,dtype=np.float64))

                
                with open ('pca_entropy.csv',mode="a") as file:
                    writer = csv.writer(file,delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
                    writer.writerow([str(total_entropy),str(0.0)])

	
	
	
	
	
	
	
	
	
	
	