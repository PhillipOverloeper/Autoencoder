from torch import nn
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as pl
import numpy as np
import csv
import torch
import math
import sys



class autoencoder(nn.Module):

    
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(nn.Linear(21,16),nn.BatchNorm1d(16),nn.PReLU(),nn.Linear(16,5),nn.BatchNorm1d(5),nn.PReLU())
        self.decoder = nn.Sequential(nn.Linear(5,16),nn.BatchNorm1d(16),nn.PReLU(),nn.Linear(16,21))
		
		
    def forward(self,x):
        y = self.encoder(x)
        x = self.decoder(y)

        return x,y	
		
		
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
	
    file_1 = 'state_action_autoencoder_3.csv' 
    file_2 = 'state_action_pca_2.csv'
    file_3 = 'state_action_baseline_2.csv'

    values_1 = get_real_values(file_1)
    values_2 = get_real_values(file_2)
    values_3 = get_real_values(file_3)
    model = torch.load('model.pt')
	
	
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
        with torch.no_grad(): 
            model.eval()
            x, vector_1 = model(torch.Tensor(entr_arr_1))
            x, vector_2 = model(torch.Tensor(entr_arr_2))
            x, vector_3 = model(torch.Tensor(entr_arr_3))
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


                for prob in numb_values:
                    p = prob/(episode*100)
                    probi = np.append(probi,p)
                    if p == 0:   
                        continue
                    tot_prob += p
                    total_entropy += -p * math.log(p, 2)
                    
                print('This is the entropy ' + str(total_entropy) + ' of the ' + str(n_dim+1))
                print(np.mean(probi,axis=0,dtype=np.float64))
                
                with open ('autoencoder_entropy.csv',mode="a") as file:
                    writer = csv.writer(file,delimiter=',',quotechar='|',quoting=csv.QUOTE_NONE)
                    writer.writerow([str(total_entropy),str(0.0)])
            
            
            
            
            
            
            
            
            
            
            
            
#        kde.fit(y)
#		
 #       probabilites = kde.score_samples(y)
  #      probabilites = np.exp(probabilites)
	#	
     #   print(probabilites)
	#
     #   total_entropy = 0
#
 #       for k in probabilites:
  #          total_entropy += -k * math.log(2, k)
	#			
     #   print(total_entropy)
		
#		for j in range(len(y)):
#			
#			a = np.append(a,y[j][0])
#			b = np.append(b,y[j][1])
#			c = np.append(c,y[j][2])
#			d = np.append(d,y[j][3])
#			e = np.append(e,y[j][4])
#			
#
#		for i in [a,b,c,d,e]:
#
#			i = np.asarray(i)
#			i = i.reshape((len(i), 1))
#			kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(i)
#			values = np.asarray([(2.5*value)/len(i) for value in range(-len(i),len(i))])
#			values = values.reshape((len(values), 1))
#			probabilites = kde.score_samples(values)
#			probabilites = np.exp(probabilites)
#
#			pl.hist(i, bins=1000, density=True)
#			pl.xlabel("Compressed value")
#			pl.ylabel("Density")
#			pl.plot(values[:], probabilites)
#			#pl.show()
#			
#			total_entropy = 0
#
#			for k in probabilites:
#				if k == 0.0:
#					continue
#				total_entropy += -k * math.log(2, k)
#				
#			print(total_entropy)
#			#name = "hidden_layer"+str(j) + "_" + str(k) + ".svg"
#			#pl.savefig("Density Estimation/" + name)
#			#pl.close()
		
			


















