C = 2.998e8 #m/s
TIME_CONST = 1000 #s T
import numpy as np
import matplotlib.pyplot as plt
import particlegenerator as partgen
import stepcalculations as stepcalc
import electronTrajectory as etraj

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
    def __str__(self):
        return '%s: Radius %f; Semi-Major Axis %f' % (self.name, self.rad, self.semimaxis)

E = Particle("Electron", 9.10938e-31, -1.60218e-19) #kg, C
P = Particle("Proton", 1.6726219e-27, 1.60218e-19) #kg, C
A = Particle("Alpha", 6.64465723e-27, 3.20436e-19) #kg, C
MARS = Planet("Mars", 3.3895e6, 2.279e11) #m, m

def testFunctions():
    print(stepcalc.didHitPlanet([1,1,1],[0,0,0],1))
    #plt = pyplot.figure(figsize=(15, 10), dpi=80, facecolor='w')
    fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3)
    fig.suptitle("Preliminary Results")
    x = np.arange(-1, 1, 0.05)
    z = np.arange(-1, 1, 0.05)
    xx, zz = np.meshgrid(x, z)
    B = np.empty((x.size,z.size,3))
    for i in range(x.size):
        for j in range(z.size):
            pt = np.array([xx[i,j],0,zz[i,j]])
            B[i,j] = stepcalc.magneticFieldAtPoint(np.array([0,0,0]), pt, 10)
    B[int(x.size/2),int(z.size/2)] = np.array([0,0,0])
    ax1.quiver(xx,zz,B[:,:,0],B[:,:,2],scale=1000, scale_units='inches')
    ax1.set_title("Magnetic Field Lines from Dipole Cross Section")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    
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
    plt.show()

def basicSim():
    simDim = np.array([MARS.semimaxis / 80, MARS.rad * 100, MARS.rad * 100])

if __name__ == "__main__":
    basicSim()
