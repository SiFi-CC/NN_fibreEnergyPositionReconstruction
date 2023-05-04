from RootToNN import Simulation, Tensor3d
import numpy as np
from numba import njit, jit


# give position in tensor based on sipm_id
@njit
def tensor_index(sipm_id):
    # determine y
    y = sipm_id // 368
    # remove third dimension
    sipm_id -= (y * 368)
    # x and z in scatterer
    if sipm_id < 112:
        x = sipm_id // 28
        z = (sipm_id % 28) + 2 
    # x and z in absorber
    else:
        x = (sipm_id + 16) // 32
        z = (sipm_id + 16) % 32
    return int(x), int(y), int(z)

# give position in matrix based on fibre_id
@njit
def matrix_index(fibre_id):
    # x and z in scatterer
    if fibre_id < 385:
        x = fibre_id // 55
        z = (fibre_id % 55) + 4 # correction
    else:
        fibre_id -= 385
        x = (fibre_id // 63) + 7
        z = fibre_id % 63
    return int(x), int(z)



def generate_training_data(simulation, output_name, event_type=None):
    '''Build and store the generated features and targets from a ROOT simulation'''

    # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to
    # fill up in z and 2 for (qdc, t)
    # Matrix dimensions: 12 * 2 - 2, no y, 7*4*2 - 1 + 8*4*2 -1, (2 for (energy, y)
    all_events_input            = -np.ones((simulation.num_entries, 12, 2, 32, 2))
    all_events_output_E         = -np.ones((simulation.num_entries, 22, 118, 1))
    all_events_output_y         = -np.ones((simulation.num_entries, 22, 118, 1))
    all_events_output           = np.concatenate((all_events_output_E, all_events_output_y),axis=3)
    print(all_events_output[0][0][0])
    print(simulation.num_entries)
    
    # iterate over events
    for idx, event in enumerate(simulation.iterate_events()):
        # load event features
        event_features = event.get_features()
        
        # make entries in tensor and saving tensor in list
        for counter, sipm_id in enumerate(event_features[2]):
            i, j, k = tensor_index(sipm_id)
            qdc = event_features[0][counter]
            t = event_features[1][counter]-np.min(event_features[1])
            if qdc <= 0:
                qdc = -1
                t   = -1
            else:
                qdc = qdc/4104.999988339841
                t   = t/10000
            all_events_input[idx][i][j][k][0] = qdc
            all_events_input[idx][i][j][k][1] = t
        
        # make entries in matrix and saving matrix in list
        for counter, fibre_id in enumerate(event_features[5]):
            n, m = matrix_index(fibre_id)
            E = event_features[3][counter]
            y = event_features[4][counter]
            if y>=-50 and y<=50 and E>0:
            	# normalize data
                y = (y+50)/100
                E = E/2.6486268267035484
            else:
                y = -1
                E = 0

            all_events_output[idx][n][m][0] = E
            all_events_output[idx][n][m][1] = y

    # save features as numpy tensors
    with open(output_name, 'wb') as f_train:
        np.savez_compressed(f_train,
                            all_events_input  = all_events_input,
                            all_events_output = all_events_output
                            )


simulation = Simulation(
    file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root")

generate_training_data(simulation=simulation, output_name='SiPM_approach_data.npz')
