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
        z = (fibre_id % 55) + 4 # matrix shift
    else: # must be in absorber
        fibre_id -= 385
        x = (fibre_id // 63) + 7
        z = fibre_id % 63
    return int(x), int(z)

# Give fibre matrix position associated with a particular combination of SiPMs
@njit
def tensor_to_matrix_position(arr):
    #arr:  Array of SiPM detections [qdc, t, idx, i, j, k]
    entry1, entry2 = arr[0], arr[1]
    # when going from scatterer to absorber, +1 has to be added to account for the missing fibres
    leap_correction = 0
    if entry1[3]>=4:
        leap_correction=-1
    m   = int(entry1[3]+entry2[3]+leap_correction)
    n   = int(entry1[5]+entry2[5])
    return m,n

# Returns list of all SiPMs coupled to the same fibre, if both SiPMs had a signal
@jit
def give_2QDCs(arr):
    #arr:  Array of SiPM detections [qdc,t,idx,i,j,k]
    stacked_qdcs=list()
    for entry1 in arr:
        for entry2 in arr:
            if entry1[4]==0 and entry2[4]==1 and entry1[0]>0 and entry2[0]>0:
                if entry1[3]-entry2[3] in [0,1] and entry1[5]-entry2[5] in [0,1]:
                    stacked_qdcs.append(np.array([entry1,entry2]))
    return stacked_qdcs

# iterates over whole datatset and returns all SiPMs coupled to the same fibre, if both SiPMs had a signal
@jit
def give_2QDCs_iterator(arr):
    all_qdc_combinations=list()
    for idx, event in enumerate(arr):
        all_qdc_combinations.append(give_2QDCs(event))
    return np.array(all_qdc_combinations)

# Generates "cubes" of SiPMs that are used as the input of a snipped
@jit(parallel=True)
def define_cube(combination_events, all_events_in, all_events_out):
    cubes=list()
    fibres=list()
    # iterate over all events
    for event in combination_events:
        # iterate over pairs of SiPMs that where both hit and are coupled to the same fibre
        for combination in event:
            sipm1, sipm2        = combination[0], combination[1]
            scatterer_x_range   = [0,1,2,3]
            absorber_x_range    = [4,5,6,7,8,9,10,11]
            # Data into cubes, using -1 if there is no signal
            # conditions applied to mind the gap between the scatterer and the absorber
            if (int(sipm1[3]) in scatterer_x_range) and (int(sipm2[3]) in scatterer_x_range):
                cube = -np.ones((3,2,3,2))
                for i in range(3):
                    for j in range(3):
                        # if SiPM is in scatterer
                        if int(sipm1[3]+i-1) in scatterer_x_range and (0 <= int(sipm1[5]+j-1) <= 31):
                            cube[i][0][j]=all_events_in[int(sipm1[2])][int(sipm1[3]+i-1)][0][int(sipm1[5]+j-1)]
                        if int(sipm2[3]+i-1) in scatterer_x_range and (0 <= int(sipm2[5]+j-1) <= 31):  
                            cube[i][1][j]=all_events_in[int(sipm2[2])][int(sipm2[3]+i-1)][1][int(sipm2[5]+j-1)]   
            elif (int(sipm1[3]) in absorber_x_range) and (int(sipm2[3]) in absorber_x_range):
                cube = -np.ones((3,2,3,2))
                for i in range(3):
                    for j in range(3):
                        # if SiPM is in scatterer
                        if int(sipm1[3]+i-1) in absorber_x_range and (0 <= int(sipm1[5]+j-1) <= 31):
                            cube[i][0][j]=all_events_in[int(sipm1[2])][int(sipm1[3]+i-1)][0][int(sipm1[5]+j-1)]
                        if int(sipm2[3]+i-1) in absorber_x_range and (0 <= int(sipm2[5]+j-1) <= 31):  
                            cube[i][1][j]=all_events_in[int(sipm2[2])][int(sipm2[3]+i-1)][1][int(sipm2[5]+j-1)]   
            # index of fibre corresponding to the center of the cube    
            m,n = tensor_to_matrix_position(combination)
            # take fibre data [E, y]
            fibre = all_events_out[int(sipm1[2])][m][n]
            cubes.append(cube)
            fibres.append(fibre)
    return np.array(cubes), np.array(fibres)
            


def generate_training_data(simulation, output_name, event_type=None):
    '''Build and store the generated features and targets from a ROOT simulation'''

    # Tensor dimensions: 1*4 + 2*4, 2 layers on y, 7*4 + 8*4 with 0 entries to
    # fill up in z and 2 for (qdc, t)
    # Matrix dimensions: 12 * 2 - 2, no y, 7*4*2 - 1 + 8*4*2 -1, (2 for (energy, y)
    all_events_input_list           = list()
    all_events_input                = -np.ones((simulation.num_entries, 12, 2, 32, 2))
    all_events_output               = -np.ones((simulation.num_entries, 22, 118, 2))

    print(simulation.num_entries)
    
    # iterate over events
    for idx, event in enumerate(simulation.iterate_events()):
        # load event features
        event_features  = event.get_features()
        # make entries in tensor and saving tensor in list
        event_input=list()
        for counter, sipm_id in enumerate(event_features[2]):
            i, j, k = tensor_index(sipm_id)
            qdc = event_features[0][counter]
            t = event_features[1][counter]-np.min(event_features[1])
            if qdc <= 0:
                qdc = -1
                t   = -1
            else:
            	# normalize data
                qdc = qdc/4104.999988339841
                t   = t/10000
            # fill data into tensor
            all_events_input[idx][i][j][k][0] = qdc
            all_events_input[idx][i][j][k][1] = t
            
            # fill list consisting of only SiPMs that where hit
            event_input.append(np.array([qdc,t,int(idx),int(i),int(j),int(k)]))
            


        # make entries in matrix and saving matrix in list
        for counter, fibre_id in enumerate(event_features[5]):
            n, m = matrix_index(fibre_id)
            E = event_features[3][counter]
            y = event_features[4][counter]
            if y>=-50 and y<=50 and E>0:
            	#normalize data
                y = (y+50)/100
                E = E/2.6486268267035484
            else:
                y = -1
                E = -1
            
            # fill data into matrix
            all_events_output[idx][n][m][0] = E
            all_events_output[idx][n][m][1] = y


        # fill list with lists of only SiPMs that where hit by event
        all_events_input_list.append(event_input)

    cubes, fibres = define_cube(give_2QDCs_iterator(all_events_input_list), all_events_input, all_events_output)

    # save features as numpy tensors
    with open(output_name, 'wb') as f_train:
        np.savez_compressed(f_train,
                            cubes  = cubes,
                            fibres = fibres
                            )


simulation = Simulation(
    file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root")
    #file_name="/net/data_g4rt/projects/SiFiCC/InputforNN/SiPMNNNewGeometry/ExampleDataFile.root")

generate_training_data(simulation=simulation, output_name='3x3E-1_gap.npz')
