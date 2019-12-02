############################################################
# Electron trajectory
############################################################

# Import matplotlib library for plotting
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

# Import numpy for numerical calculations, vectors, etc.
import numpy as np
from numpy.linalg import norm as norm

from main import MU_0
import stepcalculations as stepcalc
import particlegenerator as partgen

# simulation domain parameters
#BOX_X = 2
#BOX_Y = 1
#BOX_Z = 1

def plot_trajectory(trajectories, dim, planet, magnetPos):
    """Creates a matplotlib plot and plots a list of trajectories labeled
    by a list of masses.
    
    .. seealso:: called by :func:`main`

    :param trajectories: an array of trajectories
    :param masses:: a list of masses
    :returns: ``None``

    """

    # settings for plotting
    print("Plotting.")
    IMAGE_PATH = "trajectory.png"

    # create a plot
    plt = pyplot.figure(facecolor = 'w')
    ax = plt.add_subplot(111, projection = '3d')

    # set the title and axis labels
    ax.set_xlabel("X [meters]") 
    ax.set_ylabel("Y [meters]")
    ax.set_zlabel("Z [meters]")
    ax.set_title("Proton Trajectories")

    # for each trajectory in our array of trajectories, add a plot
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], "-",
                alpha=.7, linewidth=3)
    
    r = planet.rad
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r*np.cos(u)*np.sin(v) + planet.pos[0]
    y = r*np.sin(u)*np.sin(v) + planet.pos[1]
    z = r*np.cos(v) + planet.pos[2]
    ax.plot_wireframe(x, y, z, color="r")
    
    ax.scatter(magnetPos[0], magnetPos[1], magnetPos[2], color="g", s=100)
        
    # Define the plot limits
    pyplot.xlim(-dim[0], dim[0])
    pyplot.ylim(-dim[1], dim[1])
    ax.set_zlim(-dim[2], dim[2])

    # Draw a legend and save our plot
    pyplot.show()
    print("Saving plot to %s." % IMAGE_PATH)
    #ax.legend(loc="lower right", fancybox=True, shadow=True)
    plt.savefig(IMAGE_PATH, bbox_inches='tight')

    return None

def update_pos(position, velocity, paraVelocity, particle, B, timeStep):
    """calculates the magnetic force on the particle and moves it
    accordingly

    .. seealso: called by :func:`calculate_trajectory`

    :param position: 3D numpy array (r_x,r_y,r_z) in meters
    :param velocity: 3D numpy array (v_x,v_y,v_z) in m/s
    :param mass: scalar float in kg
    :param B: magnetic field strength, scalar float in Tesla
    :returns: the updated position and velocity (3D vectors)

    """

    # calculate the total force and accelerations on each body using
    # numpy's vector cross product
    #print(timeStep)
    #print("Beat:")
    #force = particle.charge * np.cross(velocity, B)
    
    #accel = force/particle.mass
    #print(accel)
    #position += velocity * timeStep
    energyLoss = stepcalc.synchrotron(particle, velocity, B) * timeStep
    #print(energyLoss)
    #print(velocity)
    paraVelocity = np.dot(velocity,stepcalc.unit(B))
    #if paraVelocity == 0:
        #paraVelocity += 100000 * np.random.randint(0,2) - 50000
    perpVelocity = np.sqrt(np.linalg.norm(velocity)**2 - paraVelocity**2)
    #print(paraVelocity)
    #print(perpVelocity)
    radCurv = particle.mass * perpVelocity / (particle.charge * np.linalg.norm(B))
    #print(radCurv)
    angVelocity = perpVelocity / radCurv
    angPosition = angVelocity * timeStep
    delPosXB = (np.sin(angPosition)) * radCurv
    delPosYB = np.sign(particle.charge) * (np.cos(angPosition) - 1) * radCurv
    delPosZB = paraVelocity * timeStep
    delVeloXB = np.cos(angPosition) * perpVelocity
    delVeloYB = -np.sign(particle.charge) * (np.sin(angPosition)) * perpVelocity
    delPosB = np.array([[delPosXB],[delPosYB],[delPosZB]])
    veloB = np.array([[delVeloXB],[delVeloYB],[paraVelocity]])
    #print(delVeloB)
    xBBasis = stepcalc.unit(velocity - paraVelocity * stepcalc.unit(B))
    zBBasis = stepcalc.unit(B)
    yBBasis = -np.cross(xBBasis,zBBasis)
    bBasis = np.stack([xBBasis,yBBasis,zBBasis]).T
    #print(bBasis)
    #print(delPosB)
    delPos = np.dot(bBasis,delPosB).reshape((3,))
    #print(delPos)
    velocity = np.dot(bBasis,veloB).reshape((3,))
    #print(delVelo)
    position += delPos
    #print(delPos)
    #print(delPos)
    #velocity = delVelo
    velocity -= partgen.velocityFromEnergy(energyLoss, particle.mass) * (velocity)/np.linalg.norm(velocity)
    #print(position)
    #print("Velocity:")
    #print(velocity)
    
    # FIX BY KEEPING VELOCITY AS PERP VS PARALLEL, RATHER THAN CONVERTING TO CARTESIAN, SO THAT YOU DON'T LOSE ALL X-VELOCITY
    #ALSO TRY MINIMUM RADIUS OF CURVATURE
    return position, velocity

def update_pos_boris(position, velocity, particle, B, timeStep):
    t = (particle.charge / particle.mass) * B * 0.5 * timeStep
    s = 2. * t / (1. + np.dot(t,t))
    vPrime = velocity + np.cross(velocity, t)
    vPlus = velocity + np.cross(vPrime, s)
    position += vPlus * timeStep
    return position, vPlus

def update_pos_pressure(position, velocity, particle, B, timeStep, density, magnetPos):
    """calculates the magnetic force on the particle and moves it
    accordingly

    .. seealso: called by :func:`calculate_trajectory`

    :param position: 3D numpy array (r_x,r_y,r_z) in meters
    :param velocity: 3D numpy array (v_x,v_y,v_z) in m/s
    :param mass: scalar float in kg
    :param B: magnetic field strength, scalar float in Tesla
    :returns: the updated position and velocity (3D vectors)

    """

    # calculate the total force and accelerations on each body using
    # numpy's vector cross product
    #print(timeStep)
    #print("Beat:")
    #force = particle.charge * np.cross(velocity, B)
    radR, radX, radY, radZ = stepcalc.distance(position, magnetPos)
    radUnit = stepcalc.unit(np.array([radX,radY,radZ]))
    magPressure = np.dot(B,B) / 2 * MU_0
    windPressure = density * np.dot(velocity,np.dot(velocity,radUnit))
    totPressure = magPressure + windPressure
    force = totPressure * radUnit * 1e-15
    accel = force / particle.mass
    energyLoss = stepcalc.synchrotron(particle, velocity, B) * timeStep
    position += velocity * timeStep + 0.5 * accel * timeStep ** 2
    velocity += accel * timeStep
    velocity -= partgen.velocityFromEnergy(energyLoss, particle.mass) * (velocity)/np.linalg.norm(velocity)
    #print("Velocity:")
    #print(velocity)

    # FIX BY KEEPING VELOCITY AS PERP VS PARALLEL, RATHER THAN CONVERTING TO CARTESIAN, SO THAT YOU DON'T LOSE ALL X-VELOCITY
    #ALSO TRY MINIMUM RADIUS OF CURVATURE
    return position, velocity

def calculate_trajectory(position, velocity, particle, magnetPos, mu, planet, dim, density):
    """Calculates the trajectory of the particle 

    .. seealso: called by :func:`calculate_trajectory`

    :param position: 3D vector (r_x,r_y,r_z) in meters
    :param velocity: 3D vector (v_x,v_y,v_z) in m/s
    :param mass: scalar float in kg
    :param B: magnetic field strength, scalar float in Tesla
    :returns: a numpy array of 3D vectors (np.arrays)

    """

    #print("Calculating trajectory: %.2e kg" % mass)

    # Start a list to append the positions to as we move the particle
    trajectory = [np.array(position)]
    hitPlanet = False
    # While the particle is inside the wall, update its position
    while -dim[0] < position[0] and position[0] < dim[0] and -dim[1] < position[1] and position[1] < dim[1] and -dim[2] < position[2] and position[2] < dim[2]:
        B = stepcalc.magneticFieldAtPoint(magnetPos, position, mu) + planet.iMagField
        paraVelocity = np.dot(velocity,stepcalc.unit(B))
        timeStep = stepcalc.timeStep(B)
        position, velocity = update_pos(position, velocity, paraVelocity, particle, B, timeStep)
        #position, velocity = update_pos_pressure(position, velocity, particle, B, timeStep, density, magnetPos)
        #position, velocity = update_pos_boris(position, velocity, particle, B, timeStep)
        trajectory.append(np.array(position))
        if stepcalc.didHitPlanet(position, velocity, planet.pos, planet.rad, timeStep):
            hitPlanet = True
            break
    #print(np.array(trajectory))

    return np.array(trajectory), hitPlanet

# main is the function that gets called when we run the program.
# Loops over multiples of electron mass, calculates trajectories, and
# plots them.
def trajTest():
    """ Loops over particles with integer multiples of the mass of the
    electron and shoots them through the magnetic field """

    print("Starting calculation.")

    # Magnetic field strength [Telsa]
    B = np.array([0,0,-1.0e-3])

    # Create lists to append a trajectory for each mass, 
    # and a list for masses
    trajectories = []
    masses = []

    # Loop over masses from 1 to 6 times electron mass
    for i in range(1,7):
        # Initial velocity and positionfor particle
        position = np.array([0, .5*BOX_Y, 0])
        velocity = np.array([.5*C, 0, 0])
        mass = i*MASS_E

        # calculate the list of positions the particle travels through
        trajectory = calculate_trajectory(position, velocity, mass, B)

        # add the mass and trajectory to our lists
        masses.append(mass)
        trajectories.append(np.array(trajectory))

    trajectories = np.array(trajectories)

    # Plotting each trajectory
    plot_trajectory(trajectories, masses)

# This is Python syntax which tells Python to call the function we
# created, called 'main()', only if this file was run directly, rather
# than with 'import orbital'
#if __name__ == "__main__":
#    main()

