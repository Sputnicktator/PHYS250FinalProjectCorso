import numpy as np
from main import C

def windParticleGenerator(eneDist, minEne, maxEne, minPos, maxPos):
    '''
    Energy in eV.
    '''
    t = np.linspace(minEne, maxEne, int(maxEne - minEne))
    eneProb = eneDist(t)
    maxProb = np.amax(eneProb)
    while True:
        energy = np.random.uniform(minEne,maxEne)
        probability = eneDist(energy) / maxProb
        probTest = np.random.uniform()
        if probTest < probability:
            break
    position = np.random.uniform(minPos,maxPos)
    return energy, position

def velocityFromEnergy(energy, mass):
    energy = energy * 1.60218e-19
    return C * np.sqrt(1 - (mass * (C ** 2) / (energy + mass * (C**2))) ** 2)
