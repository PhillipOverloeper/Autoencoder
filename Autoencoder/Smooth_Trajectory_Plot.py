import matplotlib.pyplot as pl
import csv
import numpy as np

def smooth_plot(episodes,iterations):

        pl.close()
        values = 0
        values = np.array([])
        files = "smooth_trajectory.csv"

        with open(files) as csvDataFile:
            csv_reader = csv.reader(csvDataFile)
            for idx,row in enumerate(csv_reader):
                if(idx==0):
                    values = np.array([row],dtype="f")
                    continue
                if (row==[]):
                    continue
                rw = np.array([row],dtype="f")
                values = np.append(values,rw)

        time = np.arange(0,iterations,0.05)
        print(values.shape)
        print(time.shape)



        pl.title("Velocity profile for endeffector")
        pl.xlabel("Time [s]")
        pl.ylabel("Velocity [radians/s]")
        pl.plot(time,values)
        #pl.show()
        name = str(episodes) + '_' + '.png'
        pl.savefig('smooth_plots/' + name)
        pl.close()
