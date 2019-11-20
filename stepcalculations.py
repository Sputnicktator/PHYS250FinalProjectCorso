import numpy as np

def distance(pt2Coord, pt1Coord):
    x = pt2Coord[0]-pt1Coord[0]
    y = pt2Coord[1]-pt1Coord[1]
    z = pt2Coord[2]-pt1Coord[2]
    return np.sqrt(x**2 + y**2 + z**2), x, y, z

def magnitude(field):
    return np.sqrt(field[0]**2 + field[1]**2 + field[2]**2)

def magneticFieldAtPoint(magCoord, ptCoord, mu):
    r, x, y, z = distance(ptCoord, magCoord)
    xB = (3 * x * z)
    yB = (3 * y * z)
    zB = (3 * z ** 2 - r ** 2)
    B = np.array([xB,yB,zB]) * mu / (r ** 5)
    return B
    #consider dipole may not be aligned with z
    
def didHitPlanet(ptCoord, planetCoord, planetR):
    r = distance(ptCoord, planetCoord)[0]
    if r <= planetR:
        return True
    else:
        return False

def timeStep(B, mu):
    time = TIME_CONST / magnitude(B)
    if time < TIME_CONST / (2 * mu):
        time = TIME_CONST / (2 * mu)
    return time
