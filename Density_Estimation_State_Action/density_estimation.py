import matplotlib.pyplot as pl
import csv
import numpy as np
from sklearn.neighbors import KernelDensity

def smooth_plot(episodes,iterations):

    for i in range(7):
        pl.close()
        values = 0
        values_ = np.array([])
        files = "text2.csv"
		

        with open(files) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for idx,row in enumerate(csv_reader):
                if(idx==0):
                    start = np.array([row],dtype="f")
                    values = start[0][i]
                    continue
                if (row==[]):
                    continue
                rw = np.array([row],dtype="f")
				
				
				
				
				
                values = np.append(values,rw[0][i])
				
		
        for j in [25,50,75,100,125,150,175,200]:
		
            start = values[0:j*100]
            print(i,j)

            start = start.reshape((len(start)), 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(start)
            starter = np.asarray([(2.5*value)/len(start) for value in range(-len(start),len(start))])
            starter = starter.reshape((len(starter), 1))

            probabilites = kde.score_samples(starter)
            probabilites = np.exp(probabilites)
            print(probabilites)

            pl.hist(start, bins=100, density=True)
            pl.xlabel("Position in Radian")
            pl.ylabel("Density")
            pl.title("Density Estimation of " + str(j) + ". episode") 
            pl.plot(starter[:], probabilites)
            #pl.show()
            name = str(i) + "_" + str(j) + ".png"
            pl.savefig("Density_Estimation_PCA_2/" + name)
            pl.close()   


        """
        pl.title("Velocity profile for endeffector")
        pl.xlabel("Time [s]")
        pl.ylabel("Velocity [radians/s]")
        pl.plot(time,values)
        #pl.show()
        name = str(episodes) + '_' + '.png'
        pl.savefig('smooth_plots/' + name)
        pl.close()
        """
smooth_plot(0,0)