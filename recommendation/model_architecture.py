from keras import models
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

class model_architecture():

    def construct(self) -> models.Sequential():
        model = models.Sequential()

        # Convolution layers

        # 1st Convolution layer
        model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(37, 100, 3), activation='relu', strides = 1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
        
        # 2nd Convolution layer
        model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(37, 100, 3), activation='relu', strides = 1))
        model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
        
        # 3rd Convolution layer
        model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=(37, 100, 3), activation='relu', strides = 2))
        model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
        
        # Fully Connected layers
        model.add(Flatten())

        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(37*100*3, )))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(3))
        model.add(Activation('softmax'))

        return model
