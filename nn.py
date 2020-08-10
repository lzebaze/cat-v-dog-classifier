from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# creates the training and validation generators for model
def get_generators(train_path, val_path, img_dim):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(train_path, target_size=(img_dim[0], img_dim[1]), batch_size=32, class_mode='binary')
    val_gen = val_datagen.flow_from_directory(val_path, target_size=(img_dim[0], img_dim[1]), batch_size=32, class_mode='binary')
    return train_gen, val_gen

# creates and compiles the neural network
def get_model(input_dim):
    model = keras.Sequential()

    # setup
    model.add(keras.layers.Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_dim))
    model.add(keras.layers.MaxPooling2D(pool_size=(4, 4), strides=4))
    model.add(keras.layers.Conv2D(32, kernel_size=(2, 2), strides=2, activation='relu'))
    model.add(keras.layers.Conv2D(64, kernel_size=(4, 4), strides=4, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(720, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # compiling
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_gen, val_gen):
    history = model.fit(train_gen, batch_size=52, epochs=45, validation_data=val_gen, steps_per_epoch=200)
    return history