"""
Goal/Purpose:  Simulating 50,000 decays of kaons into pions, and plotting the 
                phase space for all the kaons in red, and for certain 
                triggering kaons in black. The trigger is to ensure that the 
                kaons actually decay into two pions that are BOTH measured 
                by the calorimeter measuring device. 

Author: Suleman Bazai

Date: November 15, 2018
    
References: unit08_generatedecayObject python code by Professor George Gollin
    
Physics 298 owl, University of Illinois at Urbana-Champaign, 2018
"""


# Make sure to change to correct directory to be able to import
# unit08_generatedecayObject as GDO. The unit08_generatedecayObject 
# code is owned by Professor George GOllin


# import relevant libraries
import unit08_generatedecayObject as GDO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def decays(run):
    # Define and initialize variables.
    # GDO.Kdecay() is a code created by Professor George Gollin that
    # allows for [vertex, pmu_plus, pmu_minus = KDG.getdecay()] for each 
    # particle  decay simulation.
    KDG = GDO.Kdecay()
    
    
    #target to tracker distance = "run"meters passed into decays(run)
    decayruns = 50000
    numberoftriggers = 0
    #tracker 1 size
    z_tracker1 = run
    min_x1 = -.6
    max_x1 = .6
    min_y1 = -.6
    max_y1 = .6
    #tracker 2 size
    z_tracker2 = z_tracker1 + 10
    min_x2 = -.7
    max_x2 = .7
    min_y2 = -.7
    max_y2 = .7
    #calorimeter size
    z_calorimeter = z_tracker1+15
    min_xcal = -.75
    max_xcal = .75
    min_ycal = -.75
    max_ycal = .75
    #hole in calorimeter size
    z_hole = z_calorimeter
    min_xhole = -.25
    max_xhole = .25
    min_yhole = -.25
    max_yhole = .25
    
    #initialize the arrays for coordinates of phase space plot
    phase_space_theta_mrad_all = np.array([np.nan]*decayruns)
    phase_space_phi_all = np.array([np.nan]*decayruns)
    phase_space_z_all = np.array([np.nan]*decayruns)
    
    phase_space_theta_mrad_trigger = np.array([np.nan]*decayruns)
    phase_space_phi_trigger = np.array([np.nan]*decayruns)
    phase_space_z_trigger = np.array([np.nan]*decayruns)
    
    
    
    i=0
    N = 0
    #############################################################################
    # Loops through 50,000 decays and finds the pion positons in spherical coordinates
    #in order to be added to the respective arrays to plot for all decays. Also
    #checks for correctly triggered decays and adds the triggered decays into
    #a serprate set of arrays to be plotted in a different color.
    #############################################################################
    while N<decayruns:
        #get the decay vertex and momentum
        vertex, pmu_plus, pmu_minus = KDG.getdecay()
        xinitial, yinitial, zinitial = vertex[0], vertex[1], vertex[2]
        
        #calculate pion position at each tracker and calormieter
        x1position_minus = (z_tracker1-zinitial)*pmu_minus[1]/pmu_minus[3]
        x1position_plus = (z_tracker1-zinitial)*pmu_plus[1]/pmu_plus[3]
        y1position_minus = (z_tracker1-zinitial)*pmu_minus[2]/pmu_minus[3]
        y1position_plus = (z_tracker1-zinitial)*pmu_plus[2]/pmu_plus[3]
        
        x2position_minus = (z_tracker2-zinitial)*pmu_minus[1]/pmu_minus[3]
        x2position_plus = (z_tracker2-zinitial)*pmu_plus[1]/pmu_plus[3]
        y2position_minus = (z_tracker2-zinitial)*pmu_minus[2]/pmu_minus[3]
        y2position_plus = (z_tracker2-zinitial)*pmu_plus[2]/pmu_plus[3]
        
        x3position_minus = (z_calorimeter-zinitial)*pmu_minus[1]/pmu_minus[3]
        x3position_plus = (z_calorimeter-zinitial)*pmu_plus[1]/pmu_plus[3]
        y3position_minus = (z_calorimeter-zinitial)*pmu_minus[2]/pmu_minus[3]
        y3position_plus = (z_calorimeter-zinitial)*pmu_plus[2]/pmu_plus[3]
        
        #calculate theta, phi, and z in spehrical coordinates and add into arrays
        #for all decays
        phase_space_theta_mrad_all[i] = np.arccos(z_calorimeter/np.sqrt\
                    (x3position_plus**2+y3position_plus**2+z_calorimeter**2))*1000
        phase_space_phi_all[i] = np.arctan2(y3position_plus,x3position_plus)
        phase_space_z_all[i] = zinitial
        
        #statement to ensure all triggers are met
        if min_xcal<x3position_plus<max_xcal and min_ycal<y3position_plus<max_ycal and\
                min_x1<x1position_plus<max_x1 and min_y1<y1position_plus<max_y1\
                and min_x2<x2position_plus<max_x2 and min_y2<y2position_plus<max_y2\
                and (x3position_plus>max_xhole or x3position_plus<min_xhole or \
                     y3position_plus>max_yhole or y3position_plus<min_yhole)\
                and min_xcal<x3position_minus<max_xcal and \
                min_ycal<y3position_minus<max_ycal and\
                min_x1<x1position_minus<max_x1 and min_y1<y1position_minus<max_y1\
                and min_x2<x2position_minus<max_x2 and min_y2<y2position_minus<max_y2\
                and (x3position_minus>max_xhole or x3position_minus<min_xhole or \
                     y3position_minus>max_yhole or y3position_minus<min_yhole):
            numberoftriggers +=1
            #calculate theta, phi, and z in spehrical coordinates and add into arrays
            #for triggered decays
            phase_space_theta_mrad_trigger[i] = np.arccos(z_calorimeter/np.sqrt\
                    (x3position_plus**2+y3position_plus**2+z_calorimeter**2))*1000
            phase_space_phi_trigger[i] = np.arctan2(y3position_plus,x3position_plus)
            phase_space_z_trigger[i] = zinitial
            
        N += 1
        i += 1
        
        
    print("triggered events (distance to tracker 1 =", run,"meters):", numberoftriggers)
    
    #plot all decays in red, and triggered decays in black
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(0.0,15.0)
    ax.set_ylim(-3.5,3.5)
    ax.set_zlim(0,20)
    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    ax.set_zlabel("z")
    ax.set_title("Kaon decay phase space: all decays (red); triggers (black)--"+str(run)+"meters")
    ax.plot(phase_space_theta_mrad_all,phase_space_phi_all,phase_space_z_all,\
            'ro', markersize=1)
    ax.plot(phase_space_theta_mrad_trigger[:numberoftriggers],\
            phase_space_phi_trigger[:numberoftriggers],\
            phase_space_z_trigger[:numberoftriggers], 'ko', markersize=1)
     ###end of code "x" meters to 1st tracker


#run the code with different spacings to the 1st tracker
run  = 28
decays(run)

run  = 38
decays(run)

run  = 58
decays(run)

run  = 78
decays(run)