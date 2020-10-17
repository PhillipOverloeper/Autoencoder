import matplotlib.pyplot as pl
import matplotlib
import csv
import numpy as np

matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
pl.rcParams.update({'font.size': 50})


files = "loss.csv"
values = np.array([])
loss = np.array([],dtype="f")
time = np.arange(0,20)

with open(files) as csvDataFile:
    csv_reader = csv.reader(csvDataFile)
    for data in csv_reader:
        values = np.append(values,data)
for idx,data in enumerate(values):
    loss = np.append(loss,float(data))


for i in [1,2,150,200]:

	range = loss[(i*20)-20:(i*20)]
	mean = np.array([],dtype='f')
	variance = np.array([],dtype='f')
	
	for idx,j in enumerate(range):
		mean = np.append(mean,np.mean(range[0:idx+1]))
		variance = np.append(variance,np.var(range[0:idx+1]))
	
	pl.plot(time,mean,'k-')
	pl.fill_between(time,mean-variance,mean+variance)
	pl.xlabel('Epoch')
	pl.ylabel('Loss')
	axes = pl.gca()
	axes.set_ylim([0,1.4])
	pl.show()
#	pl.plot(time,range)
#	pl.xlabel('Epoch')
#	pl.ylabel('Loss')
#	pl.show()
	

#pl.title("Loss w.r.t epochs")
#pl.xlabel("Epochs")
#pl.ylabel("Loss")
#pl.plot(time,loss)
#pl.show()
