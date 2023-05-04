import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

skript = "1_1000"
path   = "fibreIO_E-1.npz"


def build_model():
    af = "tanh"
    conv= keras.models.Sequential([keras.layers.InputLayer(input_shape = (22, 118, 4)),
                                        keras.layers.Conv2D(filters = 16, kernel_size = 2, padding='same',activation="relu"),
                                        keras.layers.Conv2D(filters = 8, kernel_size = 1, padding='same',activation="relu"),
                                        keras.layers.Conv2D(filters = 4, kernel_size = 1, padding='same',activation="relu"),
                                        keras.layers.Conv2D(filters = 2, kernel_size = 1, padding='same',activation="linear"),                                  
    ])
    
                                
    conv.compile(loss="mse",
                    optimizer="nadam",
                    metrics = ["mae","binary_crossentropy","cosine_similarity"]
                    )
    return conv



with np.load(path) as data:
    input_data = data["all_events_input"]
    output_data = data["all_events_output"]

    print(input_data.shape)
    print(output_data.shape)

    # slice data
    trainset_index  = int(input_data.shape[0]*0.7)
    valset_index    = int(input_data.shape[0]*0.8)
    print(trainset_index)
    print(valset_index)
    X_train = input_data[:trainset_index]
    Y_train = output_data[:trainset_index]
    X_val   = input_data[trainset_index:valset_index]
    Y_val   = output_data[trainset_index:valset_index]
    X_test  = input_data[valset_index:]
    Y_test  = output_data[valset_index:]

    model = build_model()
    model.build(input_shape=input_data.shape) 
    model.summary()

    history = model.fit(X_train, Y_train, epochs=1000, 
                        validation_data=(X_val, Y_val),
                        callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=10)],
                        batch_size = 1000
                        )
    #model.summary()

    #evaluate model
    score = model.evaluate(X_test, Y_test, verbose = 0) 

    print('Test mse:', score[0]) 
    print('Test mae:', score[1])
    print('Test bc: ', score[2])
    print('Test cs: ', score[3])

    # summarize history for loss
    plt.figure(0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (MSE)')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.ylim(bottom = np.min(history.history["val_loss"])-0.0001, top = np.max(history.history['val_loss'][1]))
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('MSE_'+skript+'.png')
    

    # summarize history for mae
    plt.figure(1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.ylim(top = history.history['val_mae'][0])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('MAE_'+skript+'.png')

    # summarize history for bc
    plt.figure(1)
    plt.plot(history.history["binary_crossentropy"])
    plt.plot(history.history['val_binary_crossentropy'])
    plt.title('Model BinaryCrossentropy')
    plt.ylabel('BC')
    plt.xlabel('Epoch')
    plt.ylim(top = history.history['val_binary_crossentropy'][0])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('BC_'+skript+'.png')

    # summarize history for cs
    plt.figure(1)
    plt.plot(history.history['cosine_similarity'])
    plt.plot(history.history['val_cosine_similarity'])
    plt.title('Model CosineSimilarity')
    plt.ylabel('CS')
    plt.xlabel('Epoch')
    plt.ylim(top = history.history['val_cosine_similarity'][0])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('CS_'+skript+'.png')

    # save model
    model.save(skript+".h5")
