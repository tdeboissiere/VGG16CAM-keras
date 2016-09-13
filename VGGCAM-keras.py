import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os


def VGGCAM(nb_classes, num_input_channels=1024):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same"))
    model.add(AveragePooling2D((14, 14)))
    model.add(Flatten())
    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "VGGCAM"

    return model


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):

    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 224 * 224))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])


def train_VGGCAM(VGG_weight_path, nb_classes, num_input_channels=1024):
    """
    Train VGGCAM model

    args: VGG_weight_path (str) path to keras vgg16 weights
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer

    """

    # Load model
    model = VGGCAM(nb_classes)

    # Load weights
    with h5py.File(VGG_weight_path) as hw:
        for k in range(hw.attrs['nb_layers']):
            g = hw['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            if model.layers[k].name == "convolution2d_13":
                break
        print('Model loaded.')

    # Compile
    model.compile(optimizer="sgd", loss='categorical_crossentropy')

    # Train model with your data (dummy code)
    # update with your data

    # N.B. The data should be compatible with the VGG16 model style:

    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    # model.fit(X, y)

    # Save model
    model.save_weights(os.path.join('%s_weights.h5' % model.name))


def plot_classmap(VGGCAM_weight_path, img_path, label,
                  nb_classes, num_input_channels=1024, ratio=16):
    """
    Plot class activation map of trained VGGCAM model

    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
          img_path (str) path to the image for which we get the activation map
          label (int) label (0 to nb_classes-1) of the class activation map to plot
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
          ratio (int) upsampling ratio (16 * 14 = 224)

    """

    # Load and compile model
    model = VGGCAM(nb_classes, num_input_channels)
    model.load_weights(VGGCAM_weight_path)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    # Load and format data
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    # Get a copy of the original image
    im_ori = im.copy().astype(np.uint8)
    # VGG model normalisations
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))

    batch_size = 1
    classmap = get_classmap(model,
                            im.reshape(1, 3, 224, 224),
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    plt.imshow(im_ori)
    plt.imshow(classmap[0, label, :, :],
               cmap="jet",
               alpha=0.5,
               interpolation='nearest')
    plt.show()
    raw_input()
