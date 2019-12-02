import numpy as np
from main import TIME_CONST
from main import TIME_BOUND
from main import C
EPS_0 = 8.854187e-12 #F m

def distance(pt2Coord, pt1Coord):
    x = pt2Coord[0]-pt1Coord[0]
    y = pt2Coord[1]-pt1Coord[1]
    z = pt2Coord[2]-pt1Coord[2]
    return np.sqrt(x**2 + y**2 + z**2), x, y, z

def unit(field):
    return field / np.linalg.norm(field)

def magneticFieldAtPoint(magCoord, ptCoord, mu):
    r, x, y, z = distance(ptCoord, magCoord)
    if r == 0:
        return mu * (8 * np.pi / 3) * np.array([0,0,1])
    xB = (3 * x * z)
    yB = (3 * y * z)
    zB = (3 * z ** 2 - r ** 2)
    B = np.array([xB,yB,zB]) * mu / (r ** 5)
    return B
    #consider dipole may not be aligned with z
    
def didHitPlanet(ptCoord, velocity, planetCoord, planetR, timeStep):
    r, x, y, z = distance(ptCoord, planetCoord)
    radVec = np.array([x,y,z])
    velUnit = unit(velocity)
    det = (np.dot(velUnit, radVec)) ** 2 - (r ** 2 - planetR ** 2)
    if det < 0:
        return False
    else:
        velocitySep = -np.linalg.norm(velocity)*timeStep
        pt1 = -np.dot(velUnit, radVec) - np.sqrt(det)
        if pt1 > velocitySep and pt1 < 0:
            return True
        pt2 = -np.dot(velUnit, radVec) + np.sqrt(det)
        if pt2 < velocitySep and pt2 < 0:
            return True

def timeStep(B):
    mag = np.linalg.norm(B)
    if mag >= TIME_BOUND:
        time = TIME_CONST / mag
        if time < TIME_CONST:
            time = TIME_CONST
        return time
    else:
        time = TIME_CONST / TIME_BOUND
        return time

def synchrotron(particle, velocity, B):
    return ((particle.charge ** 4) * (np.dot(B,B)) / (6 * np.pi * EPS_0 * (particle.mass ** 4) * (C ** 5))) * ((particle.mass * C) ** 2 * np.dot(velocity,velocity))
