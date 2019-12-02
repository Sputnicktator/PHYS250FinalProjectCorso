import numpy as np
from main import C

def windParticleGenerator(posRanges):
    '''
    Energy in eV.
    '''
    #t = np.linspace(minEne, maxEne, int(maxEne - minEne))
    #eneProb = eneDist(t)
    #maxProb = np.amax(eneProb)
    #while True:
    #    energy = np.random.uniform(minEne,maxEne)
    #    probability = eneDist(energy) / maxProb
    #    probTest = np.random.uniform()
    #    if probTest < probability:
    #        break
    position = []
    for range in posRanges:
        position.append(np.random.uniform(range[0],range[1]))
    return position

def velocityFromEnergy(energy, mass):
    energy = energy * 1.60218e-19
    return C * np.sqrt(1 - (mass * (C ** 2) / (energy + mass * (C**2))) ** 2)
