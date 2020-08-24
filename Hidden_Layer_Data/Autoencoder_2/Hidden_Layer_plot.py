import matplotlib.pyplot as pl
import csv
import numpy as np
from sklearn.neighbors import KernelDensity
#[50,100,150,200,250,350,400,540,499]

for j in [150]:
    files = "hidden_layer" + str(j) + ".csv"
    print(j)
    x = []
    y = []
    z = []
    a = []
    b = []
    value = 0


    with open(files) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for idx,row in enumerate(csv_reader):
            if idx >= 0 and row != []:
                for  data in row:
                    array = []
                    string = data.replace("[","")
                    string = string.replace("]","")
                    string = string.split(" ")
                    for item in string:
                        if item != '':
                            array.append(float(item))
                    for idx,item in enumerate(array):
                        if idx == 0:
                            x.append(item)
                        elif idx == 1:
                            y.append(item)
                        elif idx == 2:
                            z.append(item)
                        elif idx == 3:
                            a.append(item)
                        elif idx == 4:
                            b.append(item)



    for k,i in enumerate([x,y,z,a,b]):
        pl.close()
        print(k)
        i = np.asarray(i)
        i = i.reshape((len(i), 1))
        kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(i)
        values = np.asarray([(2.5*value)/len(i) for value in range(-len(i),len(i))])
        values = values.reshape((len(values), 1))
        probabilites = kde.score_samples(values)
        probabilites = np.exp(probabilites)

        pl.hist(i, bins=100, density=True)
        pl.xlabel("Compressed value")
        pl.ylabel("Density")
        pl.title("Density Estimation of " + str(j+1) + ". episode") 
        pl.plot(values[:], probabilites)
        #pl.show()
        name = "hidden_layer"+str(j) + "_" + str(k) + ".png"
        pl.savefig("Density Estimation/" + name)
        pl.close()

"""
ked = []

with open(files)as csvDataFile:
    csv_reader = csv.reader(csvDataFile)
    for idx,row in enumerate(csv_reader):
        if row != []:
            bla = np.array([])
            for idy,data in enumerate(row):
                arrays = []
                string = data.replace("[","")
                string = string.replace("]","")
                string = string.split(" ")
                for item in string:
                    if item != '':
                        arrays.append(float(item))
                data = np.array([arrays])
                if idy == 0:
                    bla = data
                else:
                    bla = np.concatenate((bla,data))
            if idx == 0:
                print(bla)
                kde = KernelDensity(kernel='gaussian',bandwidth=0.01).fit(bla)
                ked = kde.score_samples(bla)
                pl.fill_between(bla, np.exp(ked), alpha=0.5)
                pl.plot(bla, np.full_like(bla, -0.001), '|k', markeredgewidth=1)
            else:
                pass
            

        

"""
    

"""
x = np.array(x)
t = np.linspace(-5,5,1000)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:,None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(t[:, None])

pl.xlabel("Encoded value")
pl.ylabel("Density")
pl.title("Density Estimation for 1. dimension after 1. episode")
pl.fill_between(t, np.exp(logprob), alpha=0.5)
pl.plot(x, np.full_like(x, -0.001), '|k', markeredgewidth=1)
pl.ylim(-0.02, 0.5)
pl.show()
"""
