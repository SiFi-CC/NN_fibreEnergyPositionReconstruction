import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

path        = "fibreIO_E-1.npz"
index       = 7467
#filenames = ["stacked_autoencoder1-1_MAE_ReLu_norm_TanH","lighter5","dNN1-3","convolutional_autoencoder3_TanH"]
filenames = ["cae1"]

data = np.load(path)
input_data  = data["all_events_input"]
output_data = data["all_events_output"]

# slice data
trainset_index  = int(input_data.shape[0]*0.7)
valset_index    = int(input_data.shape[0]*0.8)

X_test  = input_data[valset_index:]
Y_test  = output_data[valset_index:]

input_data = 0
output_data = 0

def custom_loss(penalty_weight=1):
    def custom_fn(y_true, y_pred):
        squared_loss    = tf.square(tf.abs(y_true - y_pred))
        true_std        = tfp.stats.stddev(y_true)
        pred_std        = tfp.stats.stddev(y_pred)
        var_loss        = tf.square(tf.abs(true_std - pred_std))
        return squared_loss + penalty_weight*var_loss
    return custom_fn

def make_hist(E_err, E_true, E_reco, y_err, y_true, y_reco, number, filename):
    figure, axis = plt.subplots(3, 2)

    # E err
    axis[0, 0].imshow(E_err)
    axis[0, 0].set_title("E_err")

    # For Cosine Function
    axis[0, 1].imshow(y_err)
    axis[0, 1].set_title("y_err")

    # For Tangent Function
    axis[1, 1].imshow(y_true)
    axis[1, 1].set_title("y_true")

    # For Tanh Function
    axis[2, 1].imshow(y_reco)
    axis[2, 1].set_title("y_reco")

    # For Tangent Function
    axis[1, 0].imshow(E_true)
    axis[1, 0].set_title("E_true")

    # For Tanh Function
    axis[2, 0].imshow(E_reco)
    axis[2, 0].set_title("E_reco")

    plt.savefig("images/15_12_22_"+filename+"_index="+str(number)+".png")

def evaluate_predictions(filename, cl = False):
    print(filename)

    if cl:
        model       = keras.models.load_model(filename+'.h5', custom_objects={ 'custom_fn': custom_loss() })
    else:
        model       = keras.models.load_model(filename+'.h5')
    
    model.summary()

    quantity = 50
    correction = quantity//2

    f_X_test    = model.predict(X_test[index-correction:index+correction])
    #print("Xvalue=%s, Difference=%s" % (X_test[index], abs(f_X_test[4] - Y_test[index])))

    for k in range(quantity):
        E_err = []
        p_err = []
        p_true = []
        p_reco = []
        E_true = []
        E_reco = []
        for i in range(len(f_X_test[4])):
            Ex = []
            px = []
            ptx = []
            prx = []
            Etx = []
            Erx = []
            for j in range(len(f_X_test[4][i])):
                Ex.append(abs(f_X_test[k][i][j][0] - Y_test[index-correction+k][i][j][0]))
                px.append(abs(f_X_test[k][i][j][1] - Y_test[index-correction+k][i][j][1]))
                ptx.append(Y_test[index-correction+k][i][j][1])
                prx.append(f_X_test[k][i][j][1])
                Etx.append(Y_test[index-correction+k][i][j][0])
                Erx.append(f_X_test[k][i][j][0])
            E_err.append(np.array(Ex))
            p_err.append(np.array(px))
            p_true.append(np.array(ptx))
            p_reco.append(np.array(prx))
            E_true.append(np.array(Etx))
            E_reco.append(np.array(Erx))
        E_err = np.array(E_err)
        p_err = np.array(p_err)
        p_true = np.array(p_true)
        p_reco = np.array(p_reco)+1
        E_true = np.array(E_true)
        E_reco = np.array(E_reco)

        make_hist(E_err, E_true, E_reco, p_err, p_true, p_reco, index-correction+k, filename)

for i,filename in enumerate(filenames):
    evaluate_predictions(filename)#,cl[i])
