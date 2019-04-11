###################################################################

# This file is unit08_generatedecayObject.py. It generate a kaon decay.

# Energies are in GeV, momenta in GeV/c,masses in GeV/c**2
# distances are in meters and times in seconds.


# George Gollin, University of Illinois, March 26, 2017

import numpy as np

class Kdecay:
    
    ##########################################################################
    
    # let's import numpy here so argument handling in the function calls
    # can use numpy stuff in defining default values.
    import numpy as np

    # here is the "class constructor." use it to intialize various parameters.
    # it is called automatically whenever you instantiate (create an instance of)
    # an object of this class.
    
    def __init__(self):
        
        # import library
        import numpy as np
        
        # print a message about instantiating the detector
        print("\nNow instantiating the kaon decay generator.")
        
        # exact speed of light, in m/s
        self.clight = 299792458
        
        # beam energy, kaon mass, and average kaon proper lifetime
        self.e_beam = 60.
        self.m_kaon = 0.49767
        self.tau_kaon = 8.954e-11
        
        # target position
        self.x_target = 0.
        self.y_target = 0.
        self.z_target = 0.
        
        # farthest point from target to allow kaons to decay
        self.z_max = 50.
        
        # kaon momentum, entirely along the z axis in the lab frame
        pz_kaon = np.sqrt(self.e_beam**2 - self.m_kaon**2)
        self.p_kaon = np.array([0., 0., pz_kaon])
        
        # pi+, pi- momentum in kaon rest frame
        self.p_piplus_k_rest_frame = np.array([np.nan, np.nan, np.nan])
        self.p_piminus_k_rest_frame = np.array([np.nan, np.nan, np.nan])
        
        # pi+, pi- momentum in lab frame
        self.p_piplus = np.array([np.nan, np.nan, np.nan])
        self.p_piminus = np.array([np.nan, np.nan, np.nan])

        # pi+, pi- total energy in lab frame
        self.e_piplus = np.nan
        self.e_piminus = np.nan

        # lorentz gamma and beta (v/c) for a kaon traveling down the beamline
        self.kaon_gamma = self.e_beam / self.m_kaon
        self.kaon_beta = np.sqrt(1. - (1. / self.kaon_gamma**2))

        # charged pion mass. I assume that the pions do not decay in flight.
        self.m_pion = 0.13957

        # magnitude in kaon rest frame of the momentum of a charged pion
        # from the decay of a neutral kaon   
        self.e_pion_kaon_rest_frame = self.m_kaon / 2.
        self.p_pion_kaon_rest_frame = np.sqrt(self.e_pion_kaon_rest_frame**2 - \
        self.m_pion**2)
        
        # point at which kaon decays in flight
        self.x_decay = 0.
        self.y_decay = 0.
        self.z_decay = np.nan

        # keep track of how many times we needed through the pick-a-vertex loop
        # to return a z position for a decay.        
        self.number_of_tries = 0

        # trajectories of kaon and pions
        self.k0_trajectory_start      = np.array([0., 0., 0.])
        self.k0_trajectory_stop       = np.array([0., 0., 0.])
        self.piplus_trajectory_start  = np.array([0., 0., 0.])
        self.piplus_trajectory_stop   = np.array([0., 0., 0.])
        self.piminus_trajectory_start = np.array([0., 0., 0.])
        self.piminus_trajectory_stop  = np.array([0., 0., 0.])

    ###########################################################################
    # end of class constructor __init__
    ###########################################################################
    
    # generate one kaon decay, loading various kinematic outcomes into the class
    # variables.
    
    def getdecay(self):
        
        import numpy as np

        # get the location of the kaon's decay point. this should
        # exhibit an exponential fall off moving downstream of the target. This
        # will load the class variables self.x_decay, self.y_decay, self.z_decay
        self.get_decay_vertex()
        
        # now generate the kaon decay in the kaon's rest frame. this will load the
        # class variables self.p_piplus_k_rest_frame and self.p_piminus_k_rest_frame
        self.decay_the_kaon_in_its_cm()
        
        # now transform the pion momenta to the lab frame. this will load the
        # class variables self.p_piplus and self.p_piminus.
        self.transform_to_lab()
        # we are done!
        
        # now return the following:
        #   x, y, z of decay vertex
        #   pi+ four momentum: E/c, px, py, pz
        #   pi- four momentum: E/c, px, py, pz
        
        vertex_xyz = np.array([self.x_decay, self.y_decay, self.z_decay])

        pmu_0 = self.e_piplus
        pmu_1 = self.p_piplus[0]
        pmu_2 = self.p_piplus[1]
        pmu_3 = self.p_piplus[2]
        piplus_four_momentum = np.array([pmu_0, pmu_1, pmu_2, pmu_3]) 

        pmu_0 = self.e_piminus
        pmu_1 = self.p_piminus[0]
        pmu_2 = self.p_piminus[1]
        pmu_3 = self.p_piminus[2]
        piminus_four_momentum = np.array([pmu_0, pmu_1, pmu_2, pmu_3]) 

        return vertex_xyz, piplus_four_momentum, piminus_four_momentum
        #\
        ##np.array([\
        #[self.x_decay, self.y_decay, self.z_decay], \
        #[self.e_piplus, self.p_piplus[0], self.p_piplus[1], self.p_piplus[2]], \
        #[self.e_piminus, self.p_piminus[0], self.p_piminus[1], self.p_piminus[2]] \
        #])

    ###########################################################################
    # end of class function generate_one_kaon_decay
    ###########################################################################        

    # get the point at which the kaon decays
    
    def get_decay_vertex(self):

        # import the random number routines
        import random as ran
        
        # initialize the x, y coordinates of the decay vertex
        self.x_decay = self.x_target
        self.y_decay = self.y_target
        
        # flag for use inside the loop
        we_have_a_decay_vertex = False
        
        # number of tries to get a vertex
        self.number_of_tries = 0

        # we expect to see an exponential decay law, so generate z flat, but then
        # throw away most of them.

        while not we_have_a_decay_vertex:

            self.number_of_z_tries = self.number_of_tries + 1
            
            # here is a trial vertex.
            trial_z = self.z_target + ran.random() * (self.z_max - self.z_target)
            
            # get the relative probability of decay at this point
            relative_probability = np.exp(-trial_z / \
            (self.kaon_beta * self.kaon_gamma * self.clight * self.tau_kaon))

            # get a random number and compare to decay probability
            if relative_probability >= ran.random():
                we_have_a_decay_vertex = True

        self.z_decay = trial_z

        # we are done!

    ###########################################################################
    # end of class function get_decay_vertex
    ########################################################################### 

    # get the pion momenta in the kaon rest frame
    
    def decay_the_kaon_in_its_cm(self):

        # import the random number routines
        import random as ran
        import numpy as np
        
        # we need to pick the theta and phi angles now for the decay. Let's
        # assume they describe the pi+.
        # we need to pick them flat in cos(theta) (between +1 and -1) and phi 
        # (between 0 and 2pi).
        
        # phi for pi+
        phi_piplus = ran.random() * 2. * np.pi
        
        # cos(theta) for pi+, from -1 to +1
        costheta_piplus = 2. * ran.random() - 1.

        # now get theta
        theta_piplus = np.arccos(costheta_piplus)
        
        # now get the momentum in the kaon rest frame.
        pz_piplus = self.p_pion_kaon_rest_frame * costheta_piplus       
        
        # momentum magnitude in the x,y plane
        ptransverse_piplus = self.p_pion_kaon_rest_frame * np.sin(theta_piplus)

        # x, y momentum
        px_piplus = ptransverse_piplus * np.cos(phi_piplus)
        py_piplus = ptransverse_piplus * np.sin(phi_piplus)
        
        # store in the class variable now.
        self.p_piplus_k_rest_frame = np.array([px_piplus, py_piplus, pz_piplus])
        
        # conservation of momentum in the cm constrains the pi-.        
        self.p_piminus_k_rest_frame = -self.p_piplus_k_rest_frame 

        # we are done!

    ###########################################################################
    # end of class function decay_the_kaon_in_its_cm
    ########################################################################### 

    # transform the pion momenta from the kaon rest frame into the lab frame
    # I assume the beam moves exactly along the z axis.
    
    def transform_to_lab(self):

        # x, y momenta are not changed when transforming to the lab frame.
        self.p_piplus[0] = self.p_piplus_k_rest_frame[0]
        self.p_piplus[1] = self.p_piplus_k_rest_frame[1]

        self.p_piminus[0] = self.p_piminus_k_rest_frame[0]
        self.p_piminus[1] = self.p_piminus_k_rest_frame[1]
        
        # lorentz transform the z components now.
        self.p_piplus[2] = self.kaon_gamma * (self.p_piplus_k_rest_frame[2] + \
        self.kaon_beta * self.e_pion_kaon_rest_frame)

        self.p_piminus[2] = self.kaon_gamma * (self.p_piminus_k_rest_frame[2] + \
        self.kaon_beta * self.e_pion_kaon_rest_frame)

        # lorentz transform the energies now.
        self.e_piplus = self.kaon_gamma * (self.e_pion_kaon_rest_frame + \
        self.kaon_beta * self.p_piplus_k_rest_frame[2])

        self.e_piminus = self.kaon_gamma * (self.e_pion_kaon_rest_frame + \
        self.kaon_beta * self.p_piminus_k_rest_frame[2])

        """
        # just for consistency and error checking... these should be close to 0.
        px_total = self.p_piplus[0] + self.p_piminus[0]
        py_total = self.p_piplus[1] + self.p_piminus[1]

        # pz...should be about 60 GeV/c. Sum the energies too.
        pz_total = self.p_piplus[2] + self.p_piminus[2]
        e_total = self.e_piplus + self.e_piminus
        """
        
        # we are done!

    ###########################################################################
    # end of class function transform_to_lab
    ###########################################################################      

###############################################################################
# end of class ParticleDetector
###############################################################################
