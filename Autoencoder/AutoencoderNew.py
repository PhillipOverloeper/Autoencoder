import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
import numpy as np
import random as rn
import sim
import sys
import math
import time
import csv
import os
import cma

#from Positions_plot import show_position
#from Absolute_Positions_Plot import show_scatter
#from Smooth_Trajectory_Plot import smooth_plot

class autoencoder(nn.Module):
    """
    This class represents the Neural Network of a
    Simple Autoencoder
    """
    
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(nn.Linear(21,16),nn.BatchNorm1d(16),nn.PReLU(),nn.Linear(16,5),nn.BatchNorm1d(5),nn.PReLU())
        self.decoder = nn.Sequential(nn.Linear(5,16),nn.BatchNorm1d(16),nn.PReLU(),nn.Linear(16,21))
    def forward(self,x,files,binary):
        """
        This method sends the input argument x
        through the Encoder and Decoder and returns
        the result
        """
        # The encoder part
        y = self.encoder(x)

        if binary == 1:
            # Save the encoder part for plotting
            with open (files,mode="a") as file:
                writer = csv.writer(file,delimiter=",")
                b = y.cpu()
                writer.writerow(b.detach().numpy())
        # The decoder part
        x = self.decoder(y)

        return x,y

def read_in_values(files,batch_size):
    """
    This function reads in the values from the csv file
    and prepares the data for the training of the model
    """

    # Initialize variables
    values = np.array([[]])
    # Open csv file and read data in np-array values
    with open(files) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for idx,row in enumerate(csv_reader):
            if (idx == 0):
                values = np.array([row],dtype="f")
                continue
            if row == []:
                continue
            rw = np.array([row],dtype="f")
            values = np.concatenate((values,rw))

    # Transform np-array to tensor
    features = torch.tensor(values)
    # Load tensor in to the dataloader
    dataloader = DataLoader(features,batch_size=batch_size,shuffle=True)

    return dataloader

def get_real_values(files, threshold):

    # Initialize variables
    values = np.array([[]])
    # Open csv file and read data in np-array values
    with open(files) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for idx,row in enumerate(csv_reader):
            if (idx == 0):
                values = np.array([row],dtype="f")
                continue
            if row == [] or idx <= 2*threshold:
                continue
            rw = np.array([row],dtype="f")
            values = np.concatenate((values,rw))

    return values
    

def train_model(model,dataloader,num_epochs,optimizer,criterion,files,episode,file,device):
    """
    Trains the autoencoder model with the given data
    """
    print(file)
    x = np.arange(0,20)
    # Initialze variable
    loss_array = np.arange(20,dtype="f")
    # Train the autoencoder several times
    # as specified in num_epochs
    model.train()
    for epoch in range(num_epochs):
        # Train the autoencoder for every data
        # in dataloader
        for data in (dataloader):
            # Shape data
            vector = data
            vector = vector.view(vector.size(0),-1)
            #vector = Variable(vector).type(FloatTensor)
            # Send data through the model
            model.zero_grad()
            output,y = model(vector.cuda(),file,1)
            # calculate loss)
            loss = criterion(output,vector.cuda())
            loss.backward()
            # Update model
            optimizer.step()
        # Save the loss of the epoch
        print(loss.item())
        loss_array[epoch] = loss.item()

    with open(files,mode="a") as file:
       writer = csv.writer(file,delimiter=",")
       writer.writerow(loss_array)

    # Plot the los w.r. the epochs
    pl.title(str(episode) + ". simulation episode")
    pl.xlabel("Epochs")
    pl.ylabel("Loss")
    pl.plot(x,loss_array)
    name = str(episode)
    pl.savefig('loss/' + name)

def start_communication():
    """
    This function attempts to connect to the
    simulation and, if successful, activates
    snychronous mode
    """

    print("Attempting to connect")

    # Close possibly open connections
    sim.simxFinish(-1)
    # Get the client ID
    clientID = sim.simxStart("127.0.0.1",19997,True,True,5000,5)
    # Check whether connection was successful or not
    if (clientID != -1):
        print("Connection established")
    else:
        print("Connections not established")
        sys.exit()

    # Enable synchronous mode
    sim.simxSynchronous(clientID,True)

    return clientID

def get_joint_handles(clientID):
    """
    This function retrieves the joint handles
    of the robot and returns them
    """

    # Get the joint handles
    joint_handles = np.array([-1,-1,-1,-1,-1,-1,-1])
    for i in range(0,7):
        returnCode,joint_handles[i] = sim.simxGetObjectHandle(clientID,"LBR_iiwa_7_R800_joint"+str(i+1),sim.simx_opmode_blocking)

    return joint_handles

def objective_function(solutions,model,state,kde,limits_pos,height):

    """
    This function serves as the objective function of the
    cma-es optimizer. It decides which proposed action to choose
    and which to discard
    """

    # Get every row and column from solutions (the proposed actions)
    restriction = np.zeros(32)
    for idx,data in enumerate(solutions):
        var = np.append(state,data)
        var = np.array([var])
        # Vector is current array with all currently checked solution-arrays
        if idx == 0:
            vector = var
        else:
            vector = np.concatenate((vector,var))

        
        # Check for restrictions
        for idy,item in enumerate(data):
            if abs(item) > 0.174533:
                restriction[idx] += abs(item) - 0.174533
            if state[idy] + item > np.radians(limits_pos[idy]) or state[idy] + item < (np.radians(limits_pos[idy]) * (-1)): 
                restriction[idx] += abs(state[idy] + item) - np.radians(limits_pos[idy])
  

    # Get the compressed version 
    solutions = torch.Tensor(vector)
    with torch.no_grad():      
        model.eval()
        x,y = model(solutions.cuda(),None,0)
        y = y.cpu()
    vector = y.detach().numpy()
    vector = np.asarray(vector)

    # Evaluate on the Density Estimator
    ked = kde.score_samples(vector)
    ked = np.exp(ked)

    # Check whether restriction si violated
    for idx,data in enumerate(ked):
        if restriction[idx] != 0:
            ked[idx] = 10 + restriction[idx]

    
    return ked



def get_new_values(clientID,limits_pos,model,state,action,kde):
    """
    This function uses the cma-es optimizer to choose
    a new action for the robot
    """
    # Vector for the z-ccordinate of the joint
    height = np.zeros(7,dtype='f')

    # Get the joint angles and the z-ccordinate of the joints
    sim.simxPauseCommunication(clientID,True)
    for i in range(0,7):
        returnCode, joint_positions[i] = sim.simxGetJointPosition(clientID,joint_handles[i],sim.simx_opmode_streaming)
        returnCode, var = sim.simxGetObjectPosition(clientID,joint_handles[i],-1,sim.simx_opmode_streaming)
        height[i] = var[2]
    sim.simxPauseCommunication(clientID,False)

    # Iniitialize the cma-optimizer with the current actions, a popualtion of 32 and a std dev of 0.25
    es = cma.CMAEvolutionStrategy(action,0.25,{'popsize':16,'maxiter':50})

    # Run the optimizer for 50 iterations
    i = 0
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, objective_function(solutions,model,state,kde,limits_pos,height))
        i += 1

    # Save the best action
    best_action = es.result[0]
    print(es.result[1])

    return joint_positions,best_action




def smooth_trajectory(clientID,joint_handles,intervals,vel_start,vel_end,iteration,files):
    """
    This function calculates a smooth trajectory between
    the start and the end velocities
    """

    # initialize variables
    vel_comb = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
    # Variable for plotting the smooth trajectory
    sm_tra = np.arange(20,dtype="f")
    sm_tra[0] = vel_start[1]
    # Number of time steps in the simulation per second
    steps = int(1000/intervals)
    # Calculate the velocity between start and end velocity
    # for all time steps for all joints
    for i in range(1,steps+1):
        # Current time step in milliseconds
        t = (i*intervals)/1000
        # Calculate velocities for current time step for all joints
        sim.simxPauseCommunication(clientID,True)
        for j in range(0,7):
            # Calculate current velocity with
            vel_comb[j] = (1-math.sin(t*(math.pi/2))**3)*vel_start[j] + (math.sin(t*(math.pi/2))**3)*vel_end[j]
            # vel_comb[j] = (1-(1/(1+math.exp(-8*t+4))))*vel_start[j] + (1/(1+math.exp(-8*t+4)))*vel_end[j]
            # Set current velocity
            sim.simxSetJointTargetVelocity(clientID,joint_handles[j],vel_comb[j],sim.simx_opmode_streaming)
        sim.simxPauseCommunication(clientID,False)
        sim.simxSynchronousTrigger(clientID)
        # Variable for plotting the smooth trajectory
        if i != steps:
            sm_tra[i] = vel_comb[6]
    # Save smooth trajectory for plotting
    with open (files,mode="a") as file:
        writers = csv.writer(file,delimiter=",")
        writers.writerow(sm_tra)

    return vel_comb

if True:
    # Constants
    CONSTRAINT = 10                                             # In degrees per second
    NUM_EPOCHS = 20
    EPISODES = 200
    ITERATIONS = 100
    LIMITS_POS = np.array([170,120,170,120,170,120,175])        # In degree per second
    LIMITS_VEL = np.array([98,98,100,130,140,180,180])          # In degree per second
    SIMULATION_TIME = 50                                        # In milliseconds
    BATCH_SIZE = 30

    print(torch.cuda.is_available())
        
    # Initialize variables
    vel_start = np.array([0,0,0,0,0,0,0])
    vel_end = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    vel_comb = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    joint_positions = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    clientID = start_communication()
    joint_handles = get_joint_handles(clientID)
    # Set up variables for autoencoder
    FloatTensor = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
    criterion = nn.MSELoss()
    zero_episode = 0
    counter = 0
    threshold = 0

    if os.path.isfile('episode.csv'):
        with open('episode.csv') as csvDataFile:
            for row in csvDataFile:
                print("That is the row",row)
                if row == [] or counter == 1:
                    print("True")
                    continue
                else:
                    print("False")
                    current_row = row
                    zero_episode = ""
                    for idx,item in enumerate(current_row):
                        print(item)
                        if idx == 0 and item != []:
                            zero_episode = item
                        elif idx > 0 and item != [] and item != ',':
                            zero_episode = zero_episode + item
                    
            
                    zero_episode = int(zero_episode)
                    counter += 1
                    
    else:
        zero_episode = 0

    # Set up the files
    file1 = "smooth_trajectory.csv"
    file2 = "text.csv"
    if os.path.isfile(file2) and zero_episode == 0:
        os.remove(file2)
    file3 = "absolute_positions.csv"
    file4 = "loss.csv"
    if os.path.isfile(file4) and zero_episode == 0:
        os.remove(file4)
    file5 = "joint_position.csv"

    if os.path.isfile('model.pt'):
        model = torch.load('model.pt')

    file8 = "all_plots.csv"
    if os.path.isfile(file8) and zero_episode == 0:
        os.remove(file8)


    kde = KernelDensity(kernel='gaussian',bandwidth=0.5)
    
    
    # One loop is one simulation and the proceeding
    # training of the autoencoder
    for episodes in range(zero_episode,EPISODES):
        if os.path.isfile(file5):
            os.remove(file5)
        file6 = "hidden_layer" + str(episodes) + ".csv"
        if os.path.isfile(file6):
            os.remove(file6)
        if os.path.isfile(file1):
            os.remove(file1)
        if os.path.isfile(file3):
            os.remove(file3)
        if os.path.isfile('episode.csv'):
            os.remove('episode.csv')
        with open('episode.csv',mode='a') as file:
            writer = csv.writer(file,delimiter=",")
            string = str(episodes)
            writer.writerow(string)
        
        
        # Start the current episode
        print(str(episodes+1) + ". episode")
        sim.simxSynchronous(clientID,True)
        sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
        
        # Variable for measuring time
        start_time = time.time()

        # One loop is an entire simulations à ITERATION seconds
        for iterations in range(ITERATIONS):
            
            print(str(iterations+1) + ". iteration")
            # Save the current values in a csv file
            joint_positions_absolute = []
            array_1 = np.append(joint_positions,vel_start)
            array_2 = np.append(array_1,vel_end)

            sim.simxPauseCommunication(clientID,True)
            for i in range(0,7):
                position = sim.simxGetObjectPosition(clientID,joint_handles[i],-1,sim.simx_opmode_streaming)
                joint_positions_absolute.append(np.array(position[1]))
            sim.simxPauseCommunication(clientID,False)

            with open (file2,mode="a") as file:
                writer = csv.writer(file,delimiter=",")
                writer.writerow(array_2)
            with open (file3,mode="a") as file:
                writer = csv.writer(file,delimiter=",")
                writer.writerow(joint_positions_absolute)
            with open (file8,mode='a') as file:
                writer = csv.writer(file,delimiter=",")
                writer.writerow(joint_positions_absolute)

            if episodes > 19:
                threshold = abs(episodes - 19) * 100
                values = get_real_values(file2,threshold)
            else:
                values = get_real_values(file2,0)

            with torch.no_grad():      
                model.eval()

                print(threshold)
                x,y = model(torch.Tensor(values).cuda(),None,0)
                print(y.shape)
                y = y.cpu()
                kde.fit(y)
                print(kde)
                
            # Variable for measuring time
            current_time = time.time()
            # Get new starting velocity
            vel_start = smooth_trajectory(clientID,joint_handles,SIMULATION_TIME,vel_start,vel_end,iterations+1,file1)
            # Save current valuses
            joint_positions,vel_end = get_new_values(clientID,LIMITS_POS,model,array_1,vel_start,kde)
            print("Start", vel_start)
            print("End",vel_end)

            print("Interval time: ",time.time()-current_time)
            for data in vel_end:
                if abs(data) >  0.174533:
                    print("THIS IS FALSE")
                    sys.exit('Failure')

            with open (file5,mode="a") as file:
                writer = csv.writer(file,delimiter=",")
                writer.writerow(joint_positions)

        print("Episode time: ",time.time()-start_time)

        file6 = "hidden_layer" + str(episodes) + ".csv"

        # Stop the current simulation
        sim.simxSynchronous(clientID,False)
        sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)

        #show_position(episodes,ITERATIONS)
        #show_scatter(episodes,ITERATIONS)
        #smooth_plot(episodes,ITERATIONS)
        
        # Train the model
        dataloader = read_in_values(file2,BATCH_SIZE)
        train_model(model,dataloader,NUM_EPOCHS,optimizer,criterion,file4,episodes+1,file6,device)
        vel_start = np.array([0,0,0,0,0,0,0])
        vel_end = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        joint_positions = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        torch.save(model,'model.pt')


    sys.exit("End of program")
        
        


















        
