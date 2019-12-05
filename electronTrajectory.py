############################################################
# This file originated from David Miller's "SampleExercises"
# GitHub repository as a file for tracking electron trajectories.
# While still retaining its old name, it has changed a lot
# since I first grabbed that file, generalizing it to other
# particles, adjustable magnetic fields, etc. Most importantly,
# it uses a different tactic for calculating the motion caused
# by the magnetic field.
# In fact, it technically can use 3, but only one is active at
# any given time.
############################################################


import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm as norm

from main import MU_0
import stepcalculations as stepcalc
import particlegenerator as partgen

def plot_trajectory(trajectories, dim, planet, magnetPos, particle, titleLabel=""):
    '''
    Takes a list of trajectory outputs and plots them on a 3D graph. Additionally marks the planet and dipole locations. Parameters:
    - trajectories, array of trajectories used as data source for this plot
    - dim, half-dimensions for the entire simulation
    - planet, the planet in the simulation
    - magnetPos, the position of the magnet in the simulation coordinates
    - particle, the particle being tested on in the simulation
    - titleLabel, just aesthetic, to keep the title organized
    '''

    # settings for plotting
    print("Plotting.")
    IMAGE_PATH = "trajectory.png"

    # create a plot
    plt = pyplot.figure(facecolor = 'w')
    ax = plt.add_subplot(111, projection = '3d')

    # set the title and axis labels
    ax.set_xlabel("X (" + r"m" + ")")
    ax.set_ylabel("Y (" + r"m" + ")")
    ax.set_zlabel("Z (" + r"m" + ")")
    title = particle.name + " Trajectories"
    if titleLabel:
        title += ": %s" %titleLabel
    ax.set_title(title)

    # for each trajectory in our array of trajectories, add a plot
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], "-",
                alpha=.7, linewidth=3)
    
    # Craft a mesh sphere which will be plot with the trajectories
    r = planet.rad
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r*np.cos(u)*np.sin(v) + planet.pos[0]
    y = r*np.sin(u)*np.sin(v) + planet.pos[1]
    z = r*np.cos(v) + planet.pos[2]
    ax.plot_wireframe(x, y, z, color="r")
    
    #Plot the position of the dipole as a point
    ax.scatter(magnetPos[0], magnetPos[1], magnetPos[2], color="g", s=100)
        
    # Define the plot limits
    pyplot.xlim(-dim[0], dim[0])
    pyplot.ylim(-dim[1], dim[1])
    ax.set_zlim(-dim[2], dim[2])
    
    ax.view_init(30, 225)
    # Draw a legend and save our plot
    pyplot.show()

    return None

def update_pos(position, velocity, particle, B, timeStep):
    '''
    This uses a rather unique method for calculating the motion of the particles through a magnetic field. Essentially, the difficulty with this simulation is that the particles experience the large scale deflection, but this deflection can occur in part due to it getting caught up orbiting around magnetic field lines by the dipole, with rather small orbits. To try to balance the two scales, I devised this method for the steps in the simulation. At each step the gyroradius is determined, and velocity perpendicular to the magnetic field is made into an angular velocity while parallel velocity is untouched. Then motion is just derived as motion around the gyroradius. Why is this beneficial? If a particle has a very short gyroradius with a rapid revolution frequency, then instead of the simulation interpreting it as a strong acceleration in one direction and letting the particle overshoot, the exact distance around the circle covered by the particle in the time step is determined, and its velocity follows. Parameters:
    - position, the current position of the particle
    - velocity, the current velocity of the particle
    - particle, the particle tested with
    - B, the magnetic field vector at the point
    - timeStep, the amount of time covered in the step
    
    Returns the new position and the new velocity
    '''

    energyLoss = stepcalc.synchrotron(particle, velocity, B) * timeStep #Synchrotron energy loss
    
    #Calculate components of velocity parallel/perpendicular to the B field. Note that we can think of this as a reformulation of the simulation into a new coordinate basis. Here we will think of the B-field pointing in the "z" direction, and the perpendicular velocity pointing in the "x" direction.
    paraVelocity = np.dot(velocity,stepcalc.unit(B))
    perpVelocity = np.sqrt(np.linalg.norm(velocity)**2 - paraVelocity**2)
    
    #Reformulate in terms of angular motion, starting by obtaining the gyroradius
    radCurv = particle.mass * perpVelocity / (particle.charge * np.linalg.norm(B))
    angVelocity = perpVelocity / radCurv
    angPosition = angVelocity * timeStep
    
    #Calculate change in position/velocity
    delPosXB = (np.sin(angPosition)) * radCurv
    delPosYB = np.sign(particle.charge) * (np.cos(angPosition) - 1) * radCurv
    delPosZB = paraVelocity * timeStep
    delVeloXB = np.cos(angPosition) * perpVelocity
    delVeloYB = -np.sign(particle.charge) * (np.sin(angPosition)) * perpVelocity
    delPosB = np.array([[delPosXB],[delPosYB],[delPosZB]])
    veloB = np.array([[delVeloXB],[delVeloYB],[paraVelocity]])
    
    #Now construct a basis transformation back into the original coordinate system of the simulation for the sake of standardization, then conduct the basis transformation
    xBBasis = stepcalc.unit(velocity - paraVelocity * stepcalc.unit(B))
    zBBasis = stepcalc.unit(B)
    yBBasis = -np.cross(xBBasis,zBBasis)
    bBasis = np.stack([xBBasis,yBBasis,zBBasis]).T
    delPos = np.dot(bBasis,delPosB).reshape((3,))
    velocity = np.dot(bBasis,veloB).reshape((3,))
    
    #Apply changes
    position += delPos
    velocity -= partgen.velocityFromEnergy(energyLoss, particle.mass) * (velocity)/np.linalg.norm(velocity)
    
    return position, velocity

def update_pos_boris(position, velocity, particle, B, timeStep):
    '''
    Alternative method for carrying out the step. The Boris method is commonly used in magnetic field simulation, and can be useful in Particle-in-Cell simulations. To maintain accuracy of the orbits, the motion of the particle is split into two steps, which allows the particle to rotate in the middle of the step and maintain a proper orbit shape assuming a sufficiently small time step. Not the official method used in the simulation, but produces very similar to results to what I've found with my own method, which supports the validity of my test. Parameters:
    - position, the current position of the particle
    - velocity, the current velocity of the particle
    - particle, the particle tested with
    - B, the magnetic field vector at the point
    - timeStep, the amount of time covered in the step
    
    Returns the new position and the new velocity
    '''
    t = (particle.charge / particle.mass) * B * 0.5 * timeStep #Vector representation of half the rotation of the particle
    s = 2. * t / (1. + np.dot(t,t)) #Scales t to maintain constant velocity
    vPrime = velocity + np.cross(velocity, t) #First half-step
    vPlus = velocity + np.cross(vPrime, s) #Second half-step
    position += vPlus * timeStep #Full change
    return position, vPlus

def update_pos_pressure(position, velocity, particle, B, timeStep, density, magnetPos):
    '''
    Employs a very different (and one with wildly different results) method for testing the effect of the dipole, by treating the motion as a fight between the radial pressure imposed by the magnetic field, and the velocity-directed pressure caused by the solar wind. It's an attempt to bridge the gap between the particle simulation and a magnetohydrodynamic one, mainly because balancing these forces can give a decent approximation for the distance to the magnetopause between the planet and the Sun. But, I feel least comfortable with this method, mainly because finding the impact of this pressure contention on the motion of the particle requires converting the pressure into a force, which in turn requires a pretty clear surface area to work over, and that is rather uncertain when it comes to these particles. Parameters:
    - position, the current position of the particle
    - velocity, the current velocity of the particle
    - particle, the particle tested with
    - B, the magnetic field vector at the point
    - timeStep, the amount of time covered in the step
    - density, the density of the solar wind
    - magnetPos, the position of the magnet in the coordinate system of the simulation
    
    Returns the new position and the new velocity
    '''
    
    #Get the solar wind pressure
    radR, radX, radY, radZ = stepcalc.distance(position, magnetPos)
    radUnit = stepcalc.unit(np.array([radX,radY,radZ]))
    windPressure = density * np.dot(velocity,np.dot(velocity,radUnit)) * stepcalc.unit(velocity)
    
    #Get the magnetic pressure
    magPressure = (np.dot(B,B) / 2 * MU_0) * radUnit
    
    #Both pressures are vectors so that they can easily be combined
    totPressure = magPressure + windPressure
    
    #Determine the force based on an assumption of a 1e-15 m^2 estimate of particle surface area
    force = totPressure * 1e-15
    accel = force / particle.mass
    energyLoss = stepcalc.synchrotron(particle, velocity, B) * timeStep
    position += velocity * timeStep + 0.5 * accel * timeStep ** 2
    velocity += accel * timeStep
    velocity -= partgen.velocityFromEnergy(energyLoss, particle.mass) * (velocity)/np.linalg.norm(velocity)
    return position, velocity

def calculate_trajectory(position, velocity, particle, magnetPos, mu, planet, dim, density):
    '''
    For a single particle, runs a simulation until it either is ejected out of the simulation dimensions or it collides with Mars, at which point it returns True on a flag to indicate a successful hit. Parameters:
    - position, the current position of the particle
    - velocity, the current velocity of the particle
    - particle, the particle tested with
    - magnetPos, the position of the magnet in the coordinate system of the simulation
    - mu, the magnetic moment multiplied by mu_0/4pi
    - planet, the planet in the simulation
    - dim, the half-dimensions of the simulation
    - density, the density of the solar wind
    
    Returns an array of positions along the trajectory, and a boolean representing whether it hit the planet or not.
    '''

    # Start a list to append the positions to as we move the particle, and set default the hitPlanet flag to False
    trajectory = [np.array(position)]
    hitPlanet = False
    
    # While the particle is inside the wall, update its position
    while -dim[0] < position[0] and position[0] < dim[0] and -dim[1] < position[1] and position[1] < dim[1] and -dim[2] < position[2] and position[2] < dim[2]:
    
        #Calculate magnetic field from the dipole and the Interplanetary Magnetic Field
        B = stepcalc.magneticFieldAtPoint(magnetPos, position, mu) + planet.iMagField
        
        #Based off of that field calculation, derive the time step
        timeStep = stepcalc.timeStep(B)
        
        #Three methods for obtaining the position/velocity updates. Explained in detail above; update_pos is default. If you want to glance at the other ones, you can switch which is uncommented
        position, velocity = update_pos(position, velocity, particle, B, timeStep)
        #position, velocity = update_pos_pressure(position, velocity, particle, B, timeStep, density, magnetPos)
        #position, velocity = update_pos_boris(position, velocity, particle, B, timeStep)
        
        #Store trajectory and check whether the particle hit the planet
        trajectory.append(np.array(position))
        if stepcalc.didHitPlanet(position, velocity, planet.pos, planet.rad, timeStep):
            hitPlanet = True
            break

    return np.array(trajectory), hitPlanet
