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

import stepcalculations as stepcalc

# simulation domain parameters
#BOX_X = 2
#BOX_Y = 1
#BOX_Z = 1

def plot_trajectory(trajectories, dim):
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
    ax.set_title("Electron Trajectories")

    # for each trajectory in our array of trajectories, add a plot
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], "-", 
                alpha=.7, linewidth=3)
        
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

def update_pos(position, velocity, particle, B, timeStep):
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
    field = B
    force = particle.charge * np.cross(velocity, field)
    
    accel = force/particle.mass

    # update the positions and velocity
    position += velocity*timeStep + .5*accel*timeStep**2
    velocity += accel*timeStep

    return position, velocity

def calculate_trajectory(position, velocity, particle, magnetPos, mu, planet, dim):
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
        B = stepcalc.magneticFieldAtPoint(magnetPos, position, mu)
        position, velocity = update_pos(position, velocity, particle, B, stepcalc.timeStep(B, mu, position, magnetPos))
        trajectory.append(np.array(position))
        if stepcalc.didHitPlanet(position, planet.pos, planet.rad):
            hitPlanet = True
            break

    return np.array(trajectory), hitPlanet

# main is the function that gets called when we run the program.
# Loops over multiples of electron mass, calculates trajectories, and
# plots them.
def trajTest():
    """ Loops over particles with integer multiples of the mass of the
    electron and shoots them through the magnetic field """

    print("Starting calculation.")

    # Magnetic field strength [Telsa]
    B = -1.0e-3 

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

