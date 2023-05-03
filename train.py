from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

batch_size = 128
num_classes = 10
epochs = 500
lr = 1e-2

train_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'train_data',
        target_size=(50, 50),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
        )

validation_generator = train_datagen.flow_from_directory(
        'validation_data',
        target_size=(50, 50),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size = batch_size)

chanDim = -1 #channels last

model = Sequential()
# first CONV => RELU => CONV => RELU => POOL layer set
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu', padding="same",
                  input_shape=(50,50,1)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# second CONV => RELU => CONV => RELU => POOL layer set
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, decay=lr/epochs),
              metrics=['accuracy'])

early_stopping = EarlyStopping(
        monitor     = 'val_loss', 
        min_delta   = 0, 
        patience    = 7, 
        mode        = 'min', 
        verbose     = 1
    )

checkpoint = ModelCheckpoint(
        filepath        = "numeric_char_classifier.h5",
        monitor         = 'val_loss', 
        verbose         = 1, 
        save_best_only  = True, 
        mode            = 'min', 
        period          = 1
    )

reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.1,
        patience = 5,
        verbose  = 1,
        mode     = 'min',
        min_delta  = 0,
        cooldown = 0,
        min_lr   = 0
    )

tensorboard = TensorBoard(
        log_dir = 'logs/8',
        histogram_freq = 0,
        write_graph = True,
        write_images = True
    )

model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.samples // batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames)/batch_size,
        callbacks=[early_stopping, checkpoint, reduce_on_plateau, tensorboard],
	workers = 8,
	max_queue_size = 16)

