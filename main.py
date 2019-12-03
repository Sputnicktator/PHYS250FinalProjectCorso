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
    Not terribly deep, but stores simple things like mass/charge.
    '''
    def __init__(self, name, mass, charge):
        self.name = name
        self.mass = mass
        self.charge = charge
    def __str__(self):
        return '%s: Mass %f; Charge %f' % (self.name, self.mass, self.charge)

class Planet:
    '''
    Sim. to above. It's just to keep things organized.
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
    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3)
    fig.suptitle("Preliminary Results")
    y = np.arange(-30000000, 30000000, 300000)
    z = np.arange(-30000000, 30000000, 300000)
    yy, zz = np.meshgrid(y, z)
    B = np.empty((y.size,z.size,3))
    for i in range(y.size):
        for j in range(z.size):
            pt = np.array([0,yy[i,j],zz[i,j]])
            B[i,j] = stepcalc.magneticFieldAtPoint(np.array([0,0,0]), pt, 1e14)
            #B[i,j] -= np.array([2e-9,0,0])
    B[int(y.size/2),int(z.size/2)] = np.array([0,0,0])
    ax1.quiver(yy,zz,B[:,:,1],B[:,:,2])#,scale = 5e3, scale_units = 'x')
    ax1.set_title("Magnetic Field Lines from Dipole Cross Section")
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("z (m)")
    '''
    def f(x):
        sigma = 200
        return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-((x-1000) ** 2)/(2 * sigma ** 2)) + 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-((x-9000) ** 2)/(2 * sigma ** 2))
    
    energies = np.array([])
    for i in range(1000):
        energies = np.append(energies, partgen.windParticleGenerator(f, 500, 10000, -1, 1)[0])
    
    ax2.hist(energies, bins = 200, density = True)
    ax2.set_title("Kinetic Energy Spectrum of Generated Particles")
    ax2.set_xlabel("K (eV)")
    #print(velocityFromEnergy(energies, MASS_E))
    ax3.hist(particlegenerator.velocityFromEnergy(energies, MASS_E), bins = 200, density = True)
    ax3.set_title("Initial Velocity Spread of Generated Particles")
    ax3.set_xlabel("V (m/s)")
    '''
    plt.show()

def simulation(mu,nParticles,particle,planet):
    halfSimDim = np.array([MARS.semimaxis / 50, MARS.rad * 10, MARS.rad * 10])
    planet.pos = np.array([halfSimDim[0] * 0.95, 0, 0])
    magnetPos = np.array([planet.pos[0]-1.082311e9, 0, 0])
    posRanges = []
    posRanges.append([-planet.rad,planet.rad])
    posRanges.append([-planet.rad,planet.rad])
    nDensity = 4 * 10000
    density = particle.mass * nDensity
    trajectories = []
    totHitPlanet = 0
    for i in range(nParticles):
        positionStart = partgen.windParticleGenerator(posRanges)
        positionEnd = partgen.windParticleGenerator(posRanges)
        partPositionEnd = np.array([planet.pos[0],positionEnd[0],positionEnd[1]])
        partPositionStart = partPositionEnd - planet.solWindVelocity * (halfSimDim[1] / np.abs(planet.solWindVelocity[1])) * 0.8
        partPositionStart += np.array([0, positionStart[0], positionStart[1]])
        velocity = planet.solWindVelocity
        trajectory, hitPlanet = etraj.calculate_trajectory(partPositionStart, velocity, particle, magnetPos, mu, planet, halfSimDim, density)
        trajectories.append(np.array(trajectory))
        if hitPlanet:
            totHitPlanet += 1
        print("Step; mu = %d" %mu)
    trajectories = np.array(trajectories)
    return trajectories, totHitPlanet
    #etraj.plot_trajectory(trajectories, halfSimDim, MARS, magnetPos)

def strengthTest(nParticles, nTests, particle, planet):
    mu = np.arange(1,10,2) * 1e7
    mu = np.append(mu, np.arange(1,10,2) * 1e8)
    mu = np.append(mu, np.arange(1,6,2) * 1e9)
    hitRatio = np.zeros((nTests, mu.size))
    for n in range(nTests):
        for m in range(mu.size):
            trajectories, totHitPlanet = simulation(mu[m],nParticles,particle,planet)
            hitRatio[n,m] = totHitPlanet / nParticles
    hitError = stats.sem(hitRatio,axis=0)
    print(hitError)
    hitRatio = np.average(hitRatio,axis=0)
    print(hitRatio)
    np.savez(os.path.join(sys.path[0], "strengthData.npz"), mu=mu, hitRatio=hitRatio, hitError=hitError)

def strengthPlot(file, fitting = True):
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
    ax.set_xlabel("Magnetic Moment (" + r'$\frac{\mu_{0}}{4\pi}$' + ")")
    ax.set_ylabel("Hit Rate (" + r'$\frac{n_{hit}}{n_{particles}}$' + ")")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    strengthTest(100, 5, P, MARS)
    file = np.load(os.path.join(sys.path[0], "strengthData.npz"))
    strengthPlot(file)
