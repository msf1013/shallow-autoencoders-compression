from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from loading import get_image_blocks
from keras.callbacks import Callback, ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt
from mssim import MultiScaleSSIM
from keras.utils import plot_model

mssim = []
code_length = 64
height = 32
width = 32
image_name = "city_640x640"

class MonitorCallback(Callback):
    def __init__(self, monitor='loss', value=0.01, verbose=0, model=None, y_true=None):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.model = model
        self.y_true = y_true

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        y_pred = self.model.predict(self.y_true, verbose=0)
        y_pred = np.squeeze(y_pred)
        mssim_c = MultiScaleSSIM(y_pred*255.0, self.y_true*255.0)
        mssim.append(mssim_c)
        print("mssim = " + str(mssim_c))

def l2_loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

def create_decoder():
    input_code = Input(shape=(code_length,), name="decoder_input")
    x = Dense(units=(height - 2)*(width - 2)*3, activation='tanh', name="decoder_1")(input_code)
    x = Reshape((height - 2, width - 2, 3), name="decoder_2")(x)
    decoded = Conv2DTranspose(filters=3, kernel_size=(3,3), kernel_initializer="glorot_normal",    activation="sigmoid", name="decoder_3")(x)
    return Model(input_code, decoded)
 
input_img = Input(shape=(height, width, 3), name="encoder_input")

# Full autoencoder
# Encoding part
x = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', name="encoder_1")(input_img)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="encoder_2")(x)
x = MaxPooling2D(pool_size=(2, 2), name="encoder_3")(x)
x = Dropout(0.25, name="encoder_4")(x)
x = Flatten(name="encoder_5")(x)
encoded = Dense(units=code_length, activation='tanh', name="encoder_6")(x)

# Decoding part
x = Dense(units=(height - 2)*(width - 2)*3, activation='tanh', name="decoder_1")(encoded)
x = Reshape((height - 2, width - 2, 3), name="decoder_2")(x)
decoded = Conv2DTranspose(filters=3, kernel_size=(3,3), kernel_initializer="glorot_normal", activation="sigmoid", name="decoder_3")(x)

autoencoder = Model(input_img, decoded)
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)
autoencoder.compile(optimizer='rmsprop', loss=l2_loss)

x_train, size = get_image_blocks(file_name=image_name+".tiff", h=height, w=width)
x_train = x_train.astype('float32') / 255.

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlystop = MonitorCallback(value=0.001, model=autoencoder, y_true=x_train)
callbacks_list = [earlystop]

# Run training
history = autoencoder.fit(x_train, x_train,
                epochs=3,
                batch_size=1,
                shuffle=True,
                callbacks=callbacks_list,
                verbose=1)

# Save autoencoder
autoencoder.save('tmp.h5')
decoder = create_decoder()
decoder.load_weights('tmp.h5', by_name=True)
decoder.save_weights('decoder.h5')

# Save architectures to image
#plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True, rankdir="LR")
#plot_model(encoder, to_file='encoder_model.png', show_shapes=True, show_layer_names=True, rankdir="LR")
#plot_model(decoder, to_file='decoder_model.png', show_shapes=True, show_layer_names=True, rankdir="LR")

# Recontstruct images with encoder subnetwork
codes = encoder.predict(x_train)
decoded_imgs = decoder.predict(codes)

v_blocks = int(size[0] / height)
h_blocks = int(size[1] / height)

fig, ax = plt.subplots(v_blocks, h_blocks, figsize=(20, 20))

newImg = Image.new('RGB', size, "black")
pixels = newImg.load()

for y in range(v_blocks):
    for x in range(h_blocks):
        decoded_img = np.zeros((height, width, 3))
        for i in range(0, height):
            for j in range(0, width):
                i_new = j
                j_new = height - i - 1

                decoded_img[i_new][j_new] = decoded_imgs[y*v_blocks + x][i][j]

        y_new = x
        x_new = v_blocks - y - 1

		# Display image reconstruction
        ax[y_new, x_new].imshow((decoded_img.astype('float32') * 255.).astype('uint8'))
        ax[y_new, x_new].get_xaxis().set_visible(False)
        ax[y_new, x_new].get_yaxis().set_visible(False)

        # Save reconstruction to image object
        for i in range(0, height):
            for j in range(0, width):
                i_new = height - j - 1
                j_new = i
                pixels[(y * height + i_new), (x * width + j_new)] = tuple((decoded_img[i][j] * 255.0).astype('uint8'))

newImg.save("result_" + image_name + ".tiff")
plt.show()

"""
# Print dimensions of decoder subnetwork
weights_list = decoder.get_weights()
for i, weights in enumerate(weights_list):
    print(weights.shape)
"""

# Show plot for training loss.
plt.plot(history.history['loss'])
plt.title('Loss in time')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(ymin=0)
plt.legend(['Train'], loc='upper right')
plt.show()

# Show plot for MS-SSIM.
plt.plot(mssim)
plt.title('MS-SSIM in time')
plt.ylabel('MS-SSIM')
plt.xlabel('Epoch')
plt.ylim(ymin=0)
plt.legend(['Train'], loc='center right')
plt.show()
