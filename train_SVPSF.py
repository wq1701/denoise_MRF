import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from keras.utils.vis_utils import plot_model
from model import *
from keras import models

# Define paths
checkpoint_path = "SVPSF_128_invcor_model.hdf5"
# data_content = sio.loadmat('E:\SVPSF_recon\SVPSF_128.mat')
data_content = sio.loadmat('./data/SVPSF_128.mat')
test_content = sio.loadmat('./data/SVPSF_128_test.mat')


# Get data
xtrain = 100 * data_content['xtrain']
xtrain = np.reshape(xtrain, [100, 128, 128, 1])

ytrain = 100 * data_content['ytrain_pr']
ytrain = np.reshape(ytrain, [100, 128, 128, 1])

xtest = 100 * data_content['xtest']
xtest = np.reshape(xtest, [100, 128, 128, 1])

xtest_vis = xtrain[0, :, :, :]
xtest_vis = np.reshape(xtest_vis, [1, 128, 128, 1])

ytest = 100 * data_content['ytest_pr']

# Define model hyperparameters
EPOCHS = 100
SPEREPOCH = 20


# Setup definitions for display
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['mae'],
             label='Val Error')
    plt.ylim([0, 500])
    plt.legend()
    plt.show()


model = unet()

# plot_model(model, to_file='model_plot.png')

# model_checkpoint = ModelCheckpoint('unet_pepa.hdf5', monitor='loss',verbose=1, save_best_only=True)
# cp_callback =  ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_best_only=True)
# cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=False,
#                               save_weights_only=True, mode='auto', period=1)

cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=False,
                              save_weights_only=True, mode='auto', period=10)

# es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.4)

callbacks_list = [cp_callback]
history = model.fit(
    xtrain, ytrain, steps_per_epoch=SPEREPOCH,
    epochs=EPOCHS, callbacks=callbacks_list, verbose=2)
print('Training done')
# print(history.history.keys())
history_save = {'loss': history.history['loss'], 'mae': history.history['mae'], 'mse': history.history['mse']}
sio.savemat('./MATLAB/history_invcor.mat', history_save)
plot_history(history)

# VISUALIZATION
plot_model(model, to_file='model_plot_invcor.png', show_shapes=True, show_layer_names=True)
# layer_outputs = [layer.output for layer in model.layers[:1]]
layer_outputs = [layer.output for layer in model.layers]

activation_model = models.Model(inputs=model.input,
                                outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(xtest_vis)

first_layer_activation = activations[29]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 0], cmap='viridis')
plt.show()

dictOfactivations = {'Layer'+str(i): activations[i] for i in range(0, len(activations))}
sio.savemat('./MATLAB/Lactivations_invcor.mat', dictOfactivations)

# TESTING
ypred = model.predict(xtest, verbose=1)
result = {'ypred_invcor': ypred, 'ytrain_deblur': ytrain}
sio.savemat('./MATLAB/SVPSF_128_invcor.mat', result)
