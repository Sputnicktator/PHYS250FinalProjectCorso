############################################################
# This is the main file for running the simulation. It contains
# global variable definitions, functions for controlling
# the simulations, as well as output/plotting handling.
# This file also contains the main code for starting tests,
# which can be tinkered with by modifying the main()
# function parameter.
############################################################
C = 2.998e8 #m/s
TIME_CONST = 0.00001 #s T
TIME_BOUND = 1e-5 #T
B_0 = 1e-9 #T
solRotVel = 2.7e-6 #rad/s
solRad = 6.9551e8 #m
solWindRadVelocity = 4e6 #m/s
alfRad = 50 * solRad #m
import numpy as np
MU_0 = np.pi * 4e-7 #H/m
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
import particlegenerator as partgen
import stepcalculations as stepcalc
import electronTrajectory as etraj
import os
import sys

class Particle:
    '''
    Stores the relevant characteristics of a particle, i.e. its mass and charge.
    '''
    def __init__(self, name, mass, charge):
        self.name = name
        self.mass = mass
        self.charge = charge
    def __str__(self):
        return '%s: Mass %f; Charge %f' % (self.name, self.mass, self.charge)

class Planet:
    '''
    Stores the relevant characteristics of a planet, i.e. its radius and semi-major axis, along with characteristics about the solar wind at the planet, i.e. its velocity and the strength of the Interplanetary Magnetic Field.
    '''
    def __init__(self, name, rad, semiMAxis):
        self.name = name
        self.rad = rad
        self.semimaxis = semiMAxis
        self.solWindVelocity = np.array([solWindRadVelocity, -solRotVel * alfRad * (alfRad / semiMAxis), 0])
        self.iMagField = np.array([B_0 * ((1.496e8 / semiMAxis) ** 2), -B_0 * (solRotVel * 1.496e8 / solWindRadVelocity) * (1.496e8 / semiMAxis), 0])
    def __str__(self):
        return '%s: Radius %f; Semi-Major Axis %f' % (self.name, self.rad, self.semimaxis)

E = Particle("Electron", 9.10938e-31, -1.60218e-19) #kg, C
P = Particle("Proton", 1.6726219e-27, 1.60218e-19) #kg, C
A = Particle("Alpha", 6.64465723e-27, 3.20436e-19) #kg, C
MARS = Planet("Mars", 3.3895e6, 2.279e11) #m, m

def testFunctions():
    '''
    A sanity check function to test the capabilities of functions I made for the project. Right now it's set to test the magneticFieldAtPoint function, by plotting the magnetic field over a lattice of points.
    '''
    plt = pyplot.figure(facecolor = 'w')
    ax = plt.add_subplot(111)
    ax.set_title("Preliminary Results")
    y = np.arange(-30000000, 30000000, 300000)
    z = np.arange(-30000000, 30000000, 300000)
    yy, zz = np.meshgrid(y, z)
    B = np.empty((y.size,z.size,3))
    for i in range(y.size):
        for j in range(z.size):
            pt = np.array([0,yy[i,j],zz[i,j]])
            B[i,j] = stepcalc.magneticFieldAtPoint(np.array([0,0,0]), pt, 1e14)
            #B[i,j] += np.array([2e-9,0,0])
    B[int(y.size/2),int(z.size/2)] = np.array([0,0,0])
    ax.quiver(yy,zz,B[:,:,1],B[:,:,2])#,scale = 5e3, scale_units = 'x')
    ax.set_title("Magnetic Field Lines from Dipole Cross Section")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    plt.show()

def simulation(mu, nParticles, particle, planet, halfSimDim, magnetPos, method = 1):
    '''
    This function will run a simulation for nParticles number of particles. The function takes additionally as parameters:
    - mu, the magnetic moment of the dipole
    - particle, the type of particle being fired
    - planet, the planet being tested on
    - halfSimDim, the half-dimensions of the 3D space being simulated in
    - magnetPos, the position of the magnet within the simulation space
    
    Returns the resulting trajectories and the count of the number of times the particles collide with the planet.
    '''
    nDensity = 4 * 10000
    density = particle.mass * nDensity #Density is only relevant for one of the methods by which the simulation could be run. It's not used for the method used on the poster, but I'm keeping the code for it
    trajectories = []
    totHitPlanet = 0
    for i in range(nParticles):
        #Generate particle
        position = partgen.windParticleGenerator(planet, halfSimDim)
        velocity = planet.solWindVelocity
        #Run test on particle
        trajectory, hitPlanet = etraj.calculate_trajectory(position, velocity, particle, magnetPos, mu, planet, halfSimDim, density, method = method)
        #Store test
        trajectories.append(np.array(trajectory))
        if hitPlanet: #If the planet were hit, record that
            totHitPlanet += 1
        print("Step; mu = %d" %mu)
    trajectories = np.array(trajectories) #Store full set of runs
    return trajectories, totHitPlanet

def strengthTest(nParticles, nTests, particle, planet, method = 1):
    '''
    Runs a batch of simulations that varies the strength of the magnetic dipole, and stores the inefficiency for each dipole moment (the ratio of successful hits to the planet to total particles fired). Takes as parameters:
    - nParticles, the number of particles for each magnetic moment test
    - nTests, the number of times a simulation is run for each magnetic
    - particle, the type of particle being fired
    - planet, the planet being tested on
    
    Saves the results to a file in the same repository as the simulation files.
    '''
    halfSimDim = np.array([planet.semimaxis / 50, planet.rad * 10, planet.rad * 10])
    planet.pos = np.array([halfSimDim[0] * 0.95, 0, 0])
    magnetPos = np.array([planet.pos[0]-1.082311e9, 0, 0])
    mu = np.arange(1,10,2) * 1e7
    mu = np.append(mu, np.arange(1,10,2) * 1e8)
    mu = np.append(mu, np.arange(1,6,2) * 1e9)
    hitRatio = np.zeros((nTests, mu.size))
    for n in range(nTests):
        for m in range(mu.size):
            trajectories, totHitPlanet = simulation(mu[m], nParticles, particle, planet, halfSimDim, magnetPos, method = method)
            hitRatio[n,m] = totHitPlanet / nParticles
    hitError = stats.sem(hitRatio,axis=0)
    print(hitError)
    hitRatio = np.average(hitRatio,axis=0)
    print(hitRatio)
    np.savez(os.path.join(sys.path[0], "strengthData.npz"), mu=mu, hitRatio=hitRatio, hitError=hitError)

def strengthPlot(file, fitting = True):
    '''
    Takes the results of a strength test and plots them as inefficiency vs. magnetic moment. Parameters:
    - file, the file from which the data's being read
    - fitting, whether we use SciPy to fit the graph
    '''
    mu = file['mu']
    hitRatio = file['hitRatio']
    hitError = file['hitError']
    x = np.linspace(mu[0],mu[-1],num=int(mu[-1]/mu[0]),endpoint=True)
    plt.rcParams.update({'font.size': 22})
    fit = interp1d(mu, hitRatio, kind='slinear')
    fig = plt.figure(facecolor = 'w')
    ax = fig.add_subplot(111)
    if fitting:
        ax.errorbar(mu,hitRatio,yerr=hitError,marker = 'o',linestyle = 'None',color = 'indigo', label = 'Data')
        ax.plot(x,fit(x),color = 'fuchsia', label = 'Fit')
    else:
        ax.errorbar(mu,hitRatio,yerr=hitError,marker = 'o',color = 'indigo', label = 'Data')
    ax.set_xscale('log')
    ax.set_title("Inefficiency of Dipole Shield")
    ax.set_xlabel("Magnetic Moment (" + r'$Am^2\frac{\mu_{0}}{4\pi}$' + ")")
    ax.set_ylabel("Hit Rate (" + r'$\frac{n_{hit}}{n_{particles}}$' + ")")
    ax.legend()
    plt.show()

def main(protocol = 1, method = 1):
    '''
    The main code for running tests. There are a number of useful things that can be done, so it is divided into "protocols," which you can control using the protocol parameter.
    '''
    ##Test function(s), currently the function is set only to plot the magnetic field of the dipole, but that is still something you can check out if you want
    if protocol == 1:
        testFunctions()
    
    ##Trajectory plotter, runs a set of basic simulations to pump out a 3D graph of trajectories
    elif protocol == 2:
        halfSimDim = np.array([MARS.semimaxis / 50, MARS.rad * 10, MARS.rad * 10])
        MARS.pos = np.array([halfSimDim[0] * 0.95, 0, 0])
        magnetPos = np.array([MARS.pos[0]-1.082311e9, 0, 0])
        trajectories = simulation(6e8, 25, P, MARS, halfSimDim, magnetPos, method = method)[0]
        etraj.plot_trajectory(trajectories, halfSimDim, MARS, magnetPos, P,titleLabel=r"$\mu=6e8$")
    
    ##Short strength test, runs many sets of simulations over a spread of magnetic moments to produce an inefficiency plot for the shield. Poor statistics but fast
    elif protocol == 3:
        strengthTest(5, 2, P, MARS, method = method)
        file = np.load(os.path.join(sys.path[0], "strengthData.npz"))
        strengthPlot(file)
    
    ##Long strength test, runs many sets of simulations over a spread of magnetic moments to produce an inefficiency plot for the shield. Good statistics but very slow
    elif protocol == 4:
        strengthTest(100, 5, P, MARS, method = method)
        file = np.load(os.path.join(sys.path[0], "strengthData.npz"))
        strengthPlot(file)
if __name__ == "__main__":
    main(protocol = 4, method = 2)
    
    

