# Author: Philippe Clement
# Date: oct 2022

import math
import numpy as np
from uproot_methods.classes.TVector3 import TVector3


class Event:
    # represents single in Event in .root file

    # list of leaves that are required from a ROOT file to properly
    # instantiate an Event object
    l_leaves = ['SiPMData.fSiPMTriggerTime',
                'SiPMData.fSiPMQDC',
                'SiPMData.fSiPMPosition',
                'SiPMData.fSiPMId',
                'FibreData.fFibreTime',
                'FibreData.fFibreEnergy',
                'FibreData.fFibrePosition',
                'FibreData.fFibreId'
                ]

    def __init__(self,
                 sipm_triggertime,
                 sipm_qdc,
                 sipm_pos,
                 sipm_id,
                 fibre_time,
                 fibre_energy,
                 fibre_pos,
                 fibre_id
                 ):

        # defines the main calues of a simulated event

        self.sipm_triggertime   = sipm_triggertime
        self.sipm_qdc           = sipm_qdc
        self.sipm_x             = sipm_pos.x
        self.sipm_y             = sipm_pos.y
        self.sipm_z             = sipm_pos.z
        self.sipm_id            = sipm_id
        self.fibre_time         = fibre_time
        self.fibre_energy       = fibre_energy
        self.fibre_x            = fibre_pos.x
        self.fibre_y            = fibre_pos.y
        self.fibre_z            = fibre_pos.z
        self.fibre_id           = fibre_id

    def get_features(self):

        # Return relevant features in array

        # Change y value to reflect structure of tensor (tensor[0] and
        # tensor[1])

        for i, y in enumerate(self.sipm_y):
            if y < 0:
                y = 0
            else:
                y = 1
            self.sipm_y[i] = y

        # Format of feature data:
        # Input: 3DTensor with indices corresponding to SiPM position and vector (qdc, t) as value
        # Output: 2DMatrix with indices corresponding to fibre position and vector
        # (E, y) as value

        features = np.array([np.array(self.sipm_qdc),
                             np.array(self.sipm_triggertime),
                             np.array(self.sipm_id, dtype = float),
                             np.array(self.fibre_energy),
                             np.array(self.fibre_y),
                             np.array(self.fibre_id, dtype=float)])

        return features
