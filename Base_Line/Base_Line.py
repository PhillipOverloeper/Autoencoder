import numpy as np
import random as rn
import sim
import sys
import time
import math
import csv
import os


def start_communicaton():

    """
    Attempts to connect to the simulation and if
    successful retrieves the client id and starts 
    the synchronous mode
    """

    print('Attempting to connect')
    
    # Close all open connections
    sim.simxFinish(-1)
    # Get the client ID
    clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    # Check whether connnection was successful or not
    if (clientID != 1):
        print('Connection established')
        # Enable synchronous mode
        sim.simxSynchronous(clientID,True)
    else: 
        # If not then end the program
        print('Connection could not be established')
        sys.exit()
        
    return clientID
        

def get_joint_handles(clientID):

    """
    Retrieves the current joint handles
    of the robot in the simulation
    """

    # Get the joint handles
    joint_handles = np.zeros(7,dtype='i')
    for i in range(7):
        returnCode,joint_handles[i] = sim.simxGetObjectHandle(clientID,'LBR_iiwa_7_R800_joint'+str(i+1),sim.simx_opmode_blocking)
        
    return joint_handles
    
   
def get_joint_position(clientID,joint_handles):

    """
    Retrieves the current joint angle position
    of the robot. Both in the internal and external
    format
    """

    # The current position of the joint 
    joint_positions = np.zeros(7)
    cartesian_joint_positions = np.array(([]))

    # Get the current position
    sim.simxPauseCommunication(clientID,True)
    for i in range(7):
        # Retrieve the current position
        returnCode,joint_positions[i] = sim.simxGetJointPosition(clientID,joint_handles[i],sim.simx_opmode_streaming)
        returnCode,cartesian_position = sim.simxGetObjectPosition(clientID,joint_handles[i],-1,sim.simx_opmode_streaming)
        cartesian_joint_positions = np.concatenate((cartesian_joint_positions,np.array(cartesian_position)))
    sim.simxPauseCommunication(clientID,False)
    
    return joint_positions,cartesian_joint_positions
    
    
def get_new_velocity(clientID,limits_pos,constraint,joint_positions):

    """
    Samples the new end velocity for the current iteration
    from a normal ditribution and checks whether it
    satisfies the constraints
    """

    # The current end velocity
    vel_end = np.zeros(7)
    # Transform constraint 
    constraint = math.radians(constraint)
    is_invalid = False
    
    # Get new end velocity and check whether it satisfies the constraints
    for i in range(7):
        # Choose a random sample from a Gaussian normal Distribution
        vel_end[i] = rn.gauss(0,constraint/math.sqrt(2))
        # Checck whether constraints are satisfied
        if (vel_end[i] > constraint):
            is_invalid = True
        elif (vel_end[i] < constraint*(-1)):
            is_invalid = True
        
        if (joint_positions[i] + vel_end[i] > math.radians(limits_pos[i]) or joint_positions[i] + vel_end[i]  < math.radians(limits_pos[i]*(-1))):
            is_invalid = True
            
    if (is_invalid):
        vel_end = get_new_velocity(clientID,limits_pos,constraint,joint_positions)
            
    return vel_end
            
     
def smooth_trajectory(clientID,joint_handles,vel_start,vel_end,simulation_time):

    """
    Interpolates a trajectory between the start and end velocity
    to enable the robot to move in a smooth trajectory
    """

    # Current velocity somewhere between vel_start and vel_end
    vel_current = np.zeros(7)
    # Number of time steps in the simulation per second (1000 milliseconds)
    steps = int(1000/simulation_time)
    # Calculate the current velocity between the start and the end of each time step in the code ('ITERATIONS' time steps, 1 second duration each) 
    # for each time step of the simulation ('steps' time steps, SIMULATION_TIME milliseconds duration)
    for i in range(1,steps+1):
        
        # Current time step in milliseconds
        t = (i*simulation_time)/1000
        # Calculate the velocity for the current time step of the simulation for each joint
        sim.simxPauseCommunication(clientID,True)
        for j in range(7):
            # Calculate current velocity
            vel_current[j] = (1-math.sin(t*(math.pi/2))**3)*vel_start[j] + (math.sin(t*(math.pi/2))**3)*vel_end[j]
            # Set the current velocity
            sim.simxSetJointTargetVelocity(clientID,joint_handles[j],vel_current[j],sim.simx_opmode_streaming)
        sim.simxPauseCommunication(clientID,False)
        sim.simxSynchronousTrigger(clientID)
        
    
    return vel_current
    
  
def save_to_csv(file_name,data):
    
    """
    Saves the specified data to the specified
    csv-file
    """
    
    # Save specified data in specified csv-file
    with open(file_name,mode='a') as file:
        writer = csv.writer(file,delimiter=',')
        writer.writerow(data)


def check_if_exist(file_name):
    
    """
    Deletes the specified file when it
    already exists
    """
    
    # When the specified file already exists, delete it
    if (os.path.isfile(file_name)):
        os.remove(file_name)
        

if True:

    # Constants
    CONSTRAINT = 10                                         # The constraint for the angular joint velocity. In degree per second
    EPISODES = 200                                          # The amount of episodes
    ITERATIONS = 100                                        # The amount of iterations per episode
    LIMITS_POS = np.array([170,120,170,120,170,120,175])    # The limits of the angulat joint position. In degree     
    SIMULATION_TIME = 50                                    # The time steps of the simulation. In milliseconds 
    
    # The client ID and the joint handles
    clientID = start_communicaton()
    joint_handles = get_joint_handles(clientID)
    # The velocity at the start of each time steps
    vel_start = np.zeros(7)                       
    # The position of the joints at each time step
    joint_positions,cartesian_joint_positions = get_joint_position(clientID,joint_handles)

    # CSV-Files to store data
    state_action = 'state_action.csv'                           # The state-action pair (Current position, starting and end velocity)
    cartesian_position = 'cartesian_joint_positions.csv'        # The joint positions in cartesian coordinates
    
    # Delete already existing files
    check_if_exist(state_action)                            
    check_if_exist(cartesian_position)   
    
    
    for episode in range(EPISODES): 
    
        # Start the current episode
        sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
        print(str(episode+1) + '. episode')
    
        for iteration in range(ITERATIONS):
        
            print(str(iteration+1) + '. iteration')
            interval_time = time.time()
            
            # Get the new end velocity
            vel_end = get_new_velocity(clientID,LIMITS_POS,CONSTRAINT,joint_positions)
            # Save the state and the action in a CSV-file
            save_to_csv(state_action,np.concatenate((np.concatenate((joint_positions,vel_start)),vel_end)))
            save_to_csv(cartesian_position,cartesian_joint_positions)
            # Calculate the smooth trajectory
            vel_start = smooth_trajectory(clientID,joint_handles,vel_start,vel_end,SIMULATION_TIME)
            # Get the joint positions
            joint_positions,cartesian_joint_positions = get_joint_position(clientID,joint_handles)
            
            print('Interval time: ' + str(time.time()-interval_time))
        
        
        sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)
        time.sleep(3)
    
    
    sys.exit('Program finished')
    
    
    
    
    
    
    
    
    
    