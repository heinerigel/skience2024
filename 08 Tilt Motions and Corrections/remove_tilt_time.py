import sys
import numpy as np

def remove_tilt_time(reciever, source, g, parallel=True, corr_up=False):
    """
    This method removes tilt contamination of horizontal
    seismometer components in the time domain
    
    .. type reciever: numpy.array
    .. param reciever: tilt contaminated seismometer recording (acc)
                        NOTE: if 'corr_up' == True: 'reciever' must be a list containing 
                        two arrays: [0]: X/Y acc, [1]: Z acc!
    .. type source: numpy.array
    .. param source: tilt recording (angle)
    .. type g: float
    .. param g: gravitational acceleration [m/s/s]
    .. type parallel: bolean
    .. param parallel: 'True' if tilt and acceleration axis are parallel
                       'False' if tilt and acceleration axis are anti parallel
    .. type corr_up: bolean
    .. param corr_up: set to True if you want to correct for up-down component, too

    return:
    .. type reciever_corr: numpy.array
    .. param: tilt subtracted seismometer recording (acc)
    """

    sig = -1
    if parallel:
        sig = 1

    if not corr_up:
        reciever_corr = reciever - sig * g * np.sin(source)
    if corr_up:
        if len(reciever) != 2:
            print("ERROR: no z-acc given")
            sys.exit(1)
        reciever_corr = reciever[0] - sig * g * np.sin(source) + sig * np.sin(source) * reciever[1]
        
    return reciever_corr

