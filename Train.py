import tensorflow
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1.0/225, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
test_gen = ImageDataGenerator(rescale=1.0/225)
valid_gen = ImageDataGenerator(rescale=1.0/225)

train_data = train_gen.flow_from_directory("train", target_size=(128,128), batch_size=32)
test_data = test_gen.flow_from_directory("test", target_size=(128,128), batch_size=32)
valid_data = valid_gen.flow_from_directory("validation", target_size=(128,128), batch_size=32)

train_data.image_shape

from keras.applications.vgg19 import VGG19

vg = VGG19(include_top=False, input_shape=(128,128,3))

from keras.models import Sequential

model = Sequential()

for layer in vg.layers:
    layer.trainable = False

model.add(vg)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(32, activation='relu'))

model.add(keras.layers.Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit_generator(generator=train_data,
                              steps_per_epoch=len(train_data)//32,
                              epochs=20,validation_data=valid_data,
                              validation_steps=len(valid_data)//32)

model.evaluate_generator(test_data)

model.save_weights('new_weights.h5')





