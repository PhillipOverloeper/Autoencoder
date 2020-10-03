from torch import nn
from sklearn.neighbors import KernelDensity
from feature_engine.discretisers import EqualWidthDiscretiser
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
	
	file = 'text.csv'
	model = torch.load('model.pt')
	kde = KernelDensity(kernel='gaussian',bandwidth=0.5)
	
	values = get_real_values(file)
	
	for idx in [50,100,150,200]:
	
		entr_arr = values[0:(idx*100)]
#		a = np.array([[]])
#		b = np.array([[]])
#		c = np.array([[]])
#		d = np.array([[]])
#		e = np.array([[]])

		with torch.no_grad():      
			model.eval()
			x,y = model(torch.Tensor(entr_arr))
			
		kde.fit(y)
		
		probabilites = kde.score_samples(y)
		probabilites = np.exp(probabilites)
		
		print(probabilites)
	
		total_entropy = 0

		for k in probabilites:
			total_entropy += -k * math.log(2, k)
				
		print(total_entropy)
		
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
		
			


















