import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from numba import njit,jit
import matplotlib

path        = "FinalDetectorVersion_RasterCoupling_OPM_38e8protons.npz"

filenames = ["l5/lighter5"]

data = np.load(path)
input_data  = data["all_events_input"]
output_data = data["all_events_output"]

# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]

# determine full width at half maximum for single peak
@jit
def fwhm(arr_true, arr_reco, div, i, filename, attribute):
    # only applied if there is more than 1 entry in the peak
    if arr_reco.shape[0] > 1:
        arr = arr_true-arr_reco
        y, bins = np.histogram(arr, 100)
        max_y   = np.max(y)
        xs = [bins[x] for x in range(y.shape[0]) if y[x] > max_y/2.0]
        plt.figure()
        plt.hist(arr,bins=100)
        plt.ylabel("count")
        plt.grid()
        plt.ylim(bottom=0)
        if attribute=="E":
            plt.xlabel(r"$E_{true}-E_{reco}$ in MeV")
            plt.savefig(filename+"_Ereso_hist_"+str(i)+"MeV.png")
        else:
            plt.xlabel(r"$y_{true}-y_{reco}$ in mm")
            plt.savefig(filename+"_yreso_hist_"+str(i)+"mm.png")
        return np.max(xs)-np.min(xs)
    else:
        return 0

# iterate over all peaks and determine fwhm and convert it to std_dev
@jit
def iterate_fwhm(arr_true, arr_reco, bins, filename, attribute):
    div         = np.max(arr_true)/bins
    div_arr     = np.arange(0, np.max(arr_true), div)
    fwhm_arr    = np.zeros(bins)
    for i in range(bins):
        if arr_true.size != 0:
            fwhm_arr[i] = fwhm(arr_true[np.where(arr_true < (i+1)*div)], arr_reco[np.where(arr_true < (i+1)*div)],div,i, filename, attribute)
            arr_reco    = arr_reco[np.where(arr_true > i*div)]
            arr_true    = arr_true[np.where(arr_true > i*div)]
    return div_arr+div/2, fwhm_arr/np.sqrt(8*np.log(2))
        
# return true and reconstructed data
@njit(parallel=True)
def give_E_y(Y_test, f_X_test):
    y_true = -np.ones(142635*22*118)
    y_reco = -np.ones(142635*22*118)
    E_true = -np.ones(142635*22*118)
    E_reco = -np.ones(142635*22*118)
    for i in range(142635):
        for j in range(22):
            for k in range(118):
                y_true[k+j*118+i*2596] = Y_test[i][j][k][1]*100  #22*118=2596
                y_reco[k+j*118+i*2596] = f_X_test[i][j][k][1]*100
                E_true[k+j*118+i*2596] = Y_test[i][j][k][0]*2.6486268267035484
                E_reco[k+j*118+i*2596] = f_X_test[i][j][k][0]*2.6486268267035484
    return [y_true,y_reco,E_true,E_reco]

# takes array of true and reco. Parameter determines how 0 values should be handled. For example: E=0 is not usable and thus excluded from further analysis.
# returns absolute error of the reconstruction as well as an array containing the errors of "no-hit"-fibres and errors of "hit"-fibres.
@njit(parallel=True)
def split_zeros(true,reco, equal_zero=False):
    if equal_zero:
        none_arr_bool   = true < 0
        some_arr_bool   = true >= 0
    else:
        none_arr_bool   = true <= 0
        some_arr_bool   = true > 0
    diff        = reco-true
    diff_none   = diff[none_arr_bool]
    diff_some   = diff[some_arr_bool]
    return [diff, diff_none, diff_some]

# analyses success of the classification task in greater detail
@jit
def true_false_stats(true, reco):
    #[E_arr,y_arr]
    zeros = np.zeros(true[0].size)
    y_reco_categorised = np.greater_equal(reco[1], zeros)
    y_true_categorised = np.greater_equal(true[1], zeros)
    E_reco_categorised = np.greater(reco[0], zeros)
    E_true_categorised = np.greater(true[0], zeros)

    y_compare_true_reco = np.equal(y_reco_categorised, y_true_categorised)
    E_compare_true_reco = np.equal(E_reco_categorised, E_true_categorised)
  
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
    for idy, y in enumerate(y_reco_categorised):
        if y == True and E_reco_categorised[idy] == True and y_true_categorised[idy]==True:
            true_positives +=1
        elif y == False and E_reco_categorised[idy] == False and y_true_categorised[idy]==False:
            true_negatives +=1
        elif y!= E_reco_categorised[idy] and y_true_categorised[idy]==True:
            false_negatives +=1
        elif y!= E_reco_categorised[idy] and y_true_categorised[idy]==False:
            false_positives +=1
            
    
    y_true_positives, y_false_positives, y_true_negatives, y_false_negatives = 0, 0, 0, 0


    for idreco, reco in np.ndenumerate(y_reco_categorised):
        if reco:
            if y_compare_true_reco[idreco]:
                y_true_positives  +=1
            else:
                y_false_positives +=1
        else:
            if y_compare_true_reco[idreco]:
                y_true_negatives  +=1
            else:
                y_false_negatives +=1
                
    E_true_positives, E_false_positives, E_true_negatives, E_false_negatives = 0, 0, 0, 0
                
    for idreco, reco in np.ndenumerate(E_reco_categorised):
        if reco:
            if E_compare_true_reco[idreco]:
                E_true_positives  +=1
            else:
                E_false_positives +=1
        else:
            if E_compare_true_reco[idreco]:
                E_true_negatives  +=1
            else:
                E_false_negatives +=1
                
        output_arr = np.array([true_positives, false_positives, true_negatives, false_negatives,E_true_positives, E_false_positives, E_true_negatives, E_false_negatives, y_true_positives, y_false_positives, y_true_negatives, y_false_negatives])
    return [output_arr, output_arr/true[0].size]

# analyses success of the classification task in less detail
@jit(parallel=True)
def reduced_true_false_stats(true, reco):
    #[E_arr,y_arr]
    zeros = np.zeros(true[0].size)
    y_reco_categorised = np.greater_equal(reco[1], zeros)
    y_true_categorised = np.greater_equal(true[1], zeros)
    E_reco_categorised = np.greater(reco[0], zeros)
    E_true_categorised = np.greater(true[0], zeros)

    y_compare_true_reco = np.equal(y_reco_categorised, y_true_categorised)
    E_compare_true_reco = np.equal(E_reco_categorised, E_true_categorised)
  
    E_and_y_pos, E_and_y_neg, E_pos, E_neg, y_pos, y_neg, none_pos, none_neg = 0,0,0,0,0,0,0,0
    for i in range(true[0].size):
        if y_compare_true_reco[i] == True and E_compare_true_reco[i] == True:
            if y_reco_categorised[i] == True:
                E_and_y_pos +=1
            else:
                E_and_y_neg +=1
        elif E_compare_true_reco[i] == True and y_compare_true_reco[i] == False:
            if E_reco_categorised[i] == True:
                E_pos +=1
            else:
                E_neg +=1
        elif E_compare_true_reco[i] == False and y_compare_true_reco[i] == True:
            if y_reco_categorised[i] == True:
                y_pos +=1
            else:
                y_neg +=1
        else:
            if E_reco_categorised[i] == True:
                none_pos +=1
            else:
                none_neg +=1
        output = np.array([E_and_y_pos, E_and_y_neg, E_pos, E_neg, y_pos, y_neg, none_pos, none_neg])
    return [output, output/true[0].size*100]

@jit
def evaluate_predictions(filename):
    print(filename)
    
    
    model       = keras.models.load_model(filename+'.h5')
    
    model.summary()

    f_X_test    = model.predict(X_test)
    arrays = give_E_y(Y_test, f_X_test)
    y_true = arrays[0]
    y_reco = arrays[1]
    E_true = arrays[2]
    E_reco = arrays[3]

    all_true = np.array([E_true, y_true])
    all_reco = np.array([E_reco, y_reco])
    print(all_true.size)
    print(all_reco.size)
    # assess classification
    print("Eyp, Eyn, Ep, En, yp, yn, np, nn")
    print(reduced_true_false_stats(all_true, all_reco))
    
    


    # calculate errors
    E_diffs = split_zeros(E_true, E_reco, True)
    y_diffs = split_zeros(y_true, y_reco, True)
    
    # calculate FWHMs and resolutions
    E_bins, E_fwhms = iterate_fwhm(E_true[np.where(E_true > 0)], E_reco[np.where(E_true > 0)],100, filename, "E")
    
    E_resolutions = E_fwhms/E_bins*100
    y_bins, y_fwhms = iterate_fwhm(y_true[np.where(y_true >= 0)], y_reco[np.where(y_true >= 0)],50, filename, "y")

    fig, ax = plt.subplots()
    h = plt.hist2d(y_true, E_diffs[0], bins=[110,200], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$y_{true}$ in mm')
    plt.ylabel(r'$E_{reco} - E_{true}$ in MeV')
    plt.xlim([-110,110])
    plt.ylim([-10,10])
    plt.savefig(filename+"yt_diffE.png")



    plt.figure()
    h = plt.hist2d(E_true, y_diffs[0], bins=[130,120], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$E_{true}$ in MeV')
    plt.ylabel(r'$y_{reco} - y_{true}$ in mm')
    plt.xlim([-3,10])
    plt.ylim([-120,120])
    plt.savefig(filename+"Et_diffy.png")



    plt.figure()
    h = plt.hist2d(E_true, E_diffs[0], bins=[130,200], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$E_{true}$ in MeV')
    plt.ylabel(r'$E_{reco} - E_{true}$ in MeV')
    plt.xlim([-3,10])
    plt.ylim([-10,10])
    plt.savefig(filename+"Et_diffE.png")   

    plt.figure()
    h = plt.hist2d(y_true, y_diffs[0], bins=[110,120], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$y_{true}$ in mm')
    plt.ylabel(r'$y_{reco} - y_{true}$ in mm')
    plt.xlim([-110,110])
    plt.ylim([-120,120])
    plt.savefig(filename+"yt_diffy.png") 

    plt.figure()
    h = plt.hist2d(E_true, E_reco, bins=[200,200], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$E_{true}$ in MeV')
    plt.ylabel(r'$E_{reco}$ in MeV')
    plt.xlim([-3,10])
    plt.ylim([-10,12])
    plt.savefig(filename+"Et_Er.png") 

    plt.figure()
    h = plt.hist2d(y_true, y_reco, bins=[100,100], norm=matplotlib.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel(r'$y_{true}$ in mm')
    plt.ylabel(r'$y_{reco}$ in mm')
    plt.xlim([-110,110])
    plt.ylim([-120,120])
    plt.savefig(filename+"yt_yr.png") 

    
    #-------------------------------------------------------------------------------
    '''
    plt.figure()
    plt.hist(E_diffs[1], bins=200, log=True)
    plt.xlabel(r'$E_{reco}-E_{true}$ in MeV')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-10,10])
    plt.savefig(filename+"_nonediffEhist.png")

    plt.figure()
    plt.hist(E_diffs[2], bins=200, log=True)
    plt.xlabel(r'$E_{reco}-E_{true}$ in MeV')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-10,10])
    plt.savefig(filename+"_somediffEhist.png")

    plt.figure()
    plt.hist(E_diffs[2], bins=2000, log=False)
    plt.xlabel(r'$E_{reco}-E_{true}$ in MeV')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-0.25,0.25])
    plt.savefig(filename+"_somediffEhist_nonlog.png")

    plt.figure()
    plt.hist(y_diffs[1], bins=120, log=True)
    plt.xlabel(r'$y_{reco}-y_{true}$ in mm')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-120,120])
    plt.savefig(filename+"_nonediffyhist.png")

    plt.figure()
    plt.hist(y_diffs[2], bins=120, log=True)
    plt.xlabel(r'$y_{reco}-y_{true}$ in mm')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-120,120])
    plt.savefig(filename+"_somediffyhist.png")

    plt.figure()
    plt.hist(y_diffs[2], bins=120, log=False)
    plt.xlabel(r'$y_{reco}-y_{true}$ in mm')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-120,120])
    plt.savefig(filename+"_somediffyhist_nonlog.png")

    plt.figure()
    zero_condition = E_true>0
    plt.hist(E_diffs[0][zero_condition]/E_true[zero_condition], bins=200, log=True)
    plt.xlabel(r'$E_{reco}-E_{true}$/$E_{true}$')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-100,100])
    plt.savefig(filename+"diffErelhist.png")
    '''
    #-------------------------------------------------------------------------------

    plt.figure()
    plt.hist(y_reco, bins=120, log=True)
    plt.xlabel(r'$y_{reco}$ in mm')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-120,120])
    plt.savefig(filename+"_yrecohist.png")

    plt.figure()
    plt.hist(E_reco, bins=200, log=True)
    plt.xlabel(r'$E_{reco}$ in MeV')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-5,10])
    plt.savefig(filename+"_Erecohist.png")

    plt.figure()
    plt.hist(y_true, bins=120, log=True)
    plt.xlabel(r'$y_{true}$ in mm')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-120,120])
    plt.savefig(filename+"_ytruehist.png")

    plt.figure()
    plt.hist(E_true, bins=200, log=True)
    plt.xlabel(r'$E_{true}$ in MeV')
    plt.ylabel(r'count')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim([-5,10])
    plt.savefig(filename+"_Etruehist.png")


    plt.figure()
    plt.plot(E_bins, E_resolutions)
    plt.xlabel("Energy in MeV")
    plt.ylabel('Energy resolution in %')
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim(right=10, left = 0)
    plt.savefig(filename+"_Eresolution.png")

    plt.figure()
    plt.plot(E_bins, E_fwhms)
    plt.xlabel("Energy in MeV")
    plt.ylabel("Energy resolution in MeV")
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim(right=10, left = 0)
    plt.savefig(filename+"_Efwhm.png")


    plt.figure()
    plt.plot(y_bins, y_fwhms)
    plt.xlabel("y-position in mm")
    plt.ylabel('y-position resolution in mm')
    plt.grid()
    plt.ylim(bottom=0, top=1.1*np.max(y_fwhms))
    plt.xlim(right=100, left = 0)
    plt.savefig(filename+"_yresolution.png")
    




for i,filename in enumerate(filenames):
    print("Dateinummer: "+str(i)+' '+filename)
    evaluate_predictions(filename)
