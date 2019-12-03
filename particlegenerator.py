import numpy as np
from main import C

def windParticleGenerator(planet, halfSimDim):
    '''
    Uses an accept/reject method to get initial point.
    '''
    while True:
        y = np.random.uniform(-planet.rad,planet.rad)
        z = np.random.uniform(-planet.rad,planet.rad)
        if np.sqrt(y**2 + z**2) < planet.rad:
            break
    position = np.array([planet.pos[0],y,z])
    position -= planet.solWindVelocity * (halfSimDim[1] / np.abs(planet.solWindVelocity[1])) * 0.8
    return position

def velocityFromEnergy(energy, mass):
    energy = energy * 1.60218e-19
    return C * np.sqrt(1 - (mass * (C ** 2) / (energy + mass * (C**2))) ** 2)
