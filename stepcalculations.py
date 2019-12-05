############################################################
# This file contains all useful code for calculations that
# are performed on a step-by-step basis in the simulation.
############################################################
import numpy as np
from main import TIME_CONST
from main import TIME_BOUND
from main import C
EPS_0 = 8.854187e-12 #F m

def distance(pt2Coord, pt1Coord):
    '''
    Returns the distance between two points, as well as each component of the displacement.
    '''
    x = pt2Coord[0]-pt1Coord[0]
    y = pt2Coord[1]-pt1Coord[1]
    z = pt2Coord[2]-pt1Coord[2]
    return np.sqrt(x**2 + y**2 + z**2), x, y, z

def unit(field):
    '''
    Returns a unit vector given some vector.
    '''
    return field / np.linalg.norm(field)

def magneticFieldAtPoint(magCoord, ptCoord, mu):
    '''
    Calculates the magnetic field at a point using a dipole approximation, takes as a parameters:
    - magCoord, the coordinate location of the magnetic dipole.
    - ptCoord, the coordinate location where the field is being evaluated.
    - mu, the magnetic moment, with the mu_0/4pi absorbed into the term for simplicity.
    
    Returns the magnetic field vector.
    '''
    r, x, y, z = distance(ptCoord, magCoord)
    if r == 0: #The limit for the dipole approximation at r = 0 is this term, since it would diverge otherwise.
        return mu * (8 * np.pi / 3) * np.array([0,0,1])
    xB = (3 * x * z)
    yB = (3 * y * z)
    zB = (3 * z ** 2 - r ** 2)
    B = np.array([xB,yB,zB]) * mu / (r ** 5)
    return B
    
def didHitPlanet(ptCoord, velocity, planetCoord, planetR, timeStep):
    '''
    Evaluates whether the particle had traveled at least partially through a planet in a given time step, so that the simulation can terminate and mark a successful hit. The method was derived from the Wikipedia article "Line-sphere intersection." Parameters:
    - ptCoord, the coordinate of the particle position *after* the time step being checked
    - velocity, the velocity the particle as it was taking the time step. In my simulation, the velocity vector would not be the same throughout the time step, but by the time the particle is in the vicinity of Mars, the gyroradius should be large enough that a line approximation is reasonable
    - planetCoord, the position of the planet in the coordinate system
    - planetR, the radius of the planet
    - timeStep, the time step taken leading up to the current point
    '''
    r, x, y, z = distance(ptCoord, planetCoord) #Get the distance
    radVec = np.array([x,y,z]) #Find the radius
    velUnit = unit(velocity) #Unit vector of velocity
    det = (np.dot(velUnit, radVec)) ** 2 - (r ** 2 - planetR ** 2) #This is the determinant of a quadratic equation that checks if the point is located on the shell of the sphere, i.e. r distance away
    if det < 0: #This would imply that there is no real solution, i.e. there's no intersection
        return False
    else: #There is an intersection, but we still have to test if this intersection point occured in the time step
        velocitySep = -np.linalg.norm(velocity)*timeStep #The location of the point we theoretically started from in a line approximation of the time step
        pt1 = -np.dot(velUnit, radVec) - np.sqrt(det) #Solution 1 of the equation
        pt2 = -np.dot(velUnit, radVec) + np.sqrt(det) #Solution 2 of the equation, could be the same as 1 but likely isn't
        if pt1 > velocitySep and pt1 < 0: #Pt1 is contained within the time step
            return True
        elif pt2 > velocitySep and pt2 < 0: #Pt2 is contained within the time step, but this would likely occur only after Pt1 had so is not entirely necessary
            return True
        else: #The particle is on an interception course but did not intercept with the planet
            return False

def timeStep(B):
    '''
    To avoid wasting time when the magnetic field is weak, we have a variable time step. When the magnetic field is stronger, subtle changes can have more drastic effects, so we have a shorter time step.
    
    Takes the magnetic field as parameter, and returns the time step.
    '''
    mag = np.linalg.norm(B)
    if mag >= TIME_BOUND: #Since B falls as r^-3, the particle can be very far away and have a very weak magnetic field, so we have a field strength limit to avoid the time step going so large that the particle blows past the important locations
        time = TIME_CONST / mag #The stronger the magnetic field, the shorter the time step
        if time < TIME_CONST: #To conserve simulation time, we also limit the time step from being too short
            time = TIME_CONST #Limit occurs when magnetic field is 1T
        return time
    else:
        time = TIME_CONST / TIME_BOUND
        return time

def synchrotron(particle, velocity, B):
    '''
    Calculates the energy loss caused by acceleration of the particle through the magnetic field. Parameters:
    - particle, the particle being simulated
    - velocity, the velocity of the particle
    - B, the magnetic field
    '''
    return ((particle.charge ** 4) * (np.dot(B,B)) / (6 * np.pi * EPS_0 * (particle.mass ** 4) * (C ** 5))) * ((particle.mass * C) ** 2 * np.dot(velocity,velocity))
