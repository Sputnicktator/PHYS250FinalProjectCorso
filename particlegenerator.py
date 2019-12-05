import numpy as np
from main import C

def windParticleGenerator(planet, halfSimDim):
    '''
    Chooses a random location for the initial particle location. We're not concerned with particles that aren't on a collision course with the planet because they'll likely be deflected away anyway, and we want to know how well the shield protects the planet from potential radiation. Parameters:
    - planet, the planet in the simulation
    - halfSimDim, the half-dimensions of the simulation
    '''
    while True:
        y = np.random.uniform(-planet.rad,planet.rad)
        z = np.random.uniform(-planet.rad,planet.rad)
        if np.sqrt(y**2 + z**2) < planet.rad: #Simple accept/reject to ensure that the particle is actually generated within planet radius
            break
    position = np.array([planet.pos[0], planet.pos[1] + y, planet.pos[2] + z]) #Initially place the particle at the planet's position, with the offset given by the accept/reject
    position -= planet.solWindVelocity * (halfSimDim[1] / np.abs(planet.solWindVelocity[1])) * 0.8 #We want the particle to start far away, so we now displace this original starting position such that the particle is far, but its velocity is in line with the planet, so it initializes in a collision course
    return position

def velocityFromEnergy(energy, mass):
    '''
    For converting the energy loss from synchrotron radiation into a change in velocity. It was originally used for something in particle generation, which is why it's in here and not stepcalculations. Parameters:
        - energy, energy used in equation
        - mass, mass used in equation
    '''
    energy = energy * 1.60218e-19
    return C * np.sqrt(1 - (mass * (C ** 2) / (energy + mass * (C**2))) ** 2)
