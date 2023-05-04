# author: Philippe Clement
# date: oct 2022

import uproot
from tqdm import tqdm
import sys
from RootToNN.SiFiCC_Module import SiFiCC_Module
from RootToNN.Event import Event

class Simulation:

    # process a ROOT Simulation for SiFi-CC detection
    
    def __init__(self, file_name):
        root_file           = uproot.open(file_name)
        #self.__setup(root_file)
        self.tree           = root_file[b'Events']
        self.num_entries    = self.tree.numentries
    
    '''    
    def __setup(self, root_file):
        
        # Extract scatterer and absorber modules setup from the ROOT file
        
        setup           = root_file[b'Setup']
        self.scatterer  = SiFiCC_Module(setup['ScattererThickness_x'].array()[0],
                                        setup['ScattererThickness_y'].array()[0], 
                                        setup['ScattererThickness_z'].array()[0], 
                                        setup['ScattererPosition'].array()[0]
                                        )
        self.absorber   = SiFiCC_Module(setup['AbsorberThickness_x'].array()[0],
                                        setup['AbsorberThickness_y'].array()[0],
                                        setup['AbsorberThickness_z'].array()[0],
                                        setup['AbsorberPosition'].array()[0]
                                        )
    
    '''                                 
    def __event_at_basket(self, basket, position):
        
        # Create and return event object at a certain position from a ROOT basket of data

        event = Event(sipm_triggertime      = basket['SiPMData.fSiPMTriggerTime'][position],
                      sipm_qdc              = basket['SiPMData.fSiPMQDC'][position],
                      sipm_pos              = basket['SiPMData.fSiPMPosition'][position],
                      sipm_id               = basket['SiPMData.fSiPMId'][position],
                      fibre_time            = basket['FibreData.fFibreTime'][position],
                      fibre_energy          = basket['FibreData.fFibreEnergy'][position],
                      fibre_pos             = basket['FibreData.fFibrePosition'][position],
                      fibre_id              = basket['FibreData.fFibreId'][position]
                     )
        return event
                                        
    def iterate_events(self, basket_size=100000, desc='processing root file', bar_update_size=1000, entrystart=None):
        
        # Iterate throughout all the events within the ROOT file. Returns an event object on each step.

        total       = self.num_entries if entrystart is None else self.num_entries - entrystart
        prog_bar    = tqdm(total = total, ncols = 100, file = sys.stdout, desc = desc)
        bar_step    = 0
        
        for start, end, basket in self.tree.iterate(Event.l_leaves, entrysteps = basket_size, reportentries = True, namedecode='utf-8', entrystart = entrystart, entrystop = None):
            length  = end - start
            
            for idx in range(length):
                yield self.__event_at_basket(basket, idx)
                
                bar_step += 1
                
                if bar_step % bar_update_size == 0:
                    prog_bar.update(bar_update_size)
                
        prog_bar.update(self.num_entries % bar_update_size)
        prog_bar.close()
    
    def get_event(self, position):
        
        # Return event object at a certain position within the ROOT file

        for basket in self.tree.iterate(Event.l_leaves, entrystart=position, entrystop=position+1, 
                                        namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)
    
