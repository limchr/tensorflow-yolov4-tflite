import os
import numpy as np
import matplotlib.pyplot as plt


import tensorflow
from tensorflow.keras import models, layers, activations
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses, metrics

tensorflow.compat.v1.disable_eager_execution()







model = models.Sequential(name="traffic_nanonet")
model.add(layers.Flatten(input_shape=(100, 100, 3)))
model.add(layers.Dense(32, activation=activations.sigmoid))
model.add(layers.Dense(4, activation=activations.softmax))

# show resulting structure and number of trainable parameters
model.summary()

# define the root directory for the data and the first level of sub-directories
data_dir = '/home/chris/datasets/test/'
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')
eval_dir = os.path.join(data_dir, 'eval')
samples_dir = os.path.join(data_dir, 'samples')

# show the names of the second level sub-directories
os.listdir(train_dir)
# Keras uses these directories to separate and to name the different image classes


# a generator in Keras provides a stream of all images stored
# in a directory tree; it is an infinite stream that shuffles
# the images randomly and restarts when the image supply has
# been exhausted

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    color_mode='rgb',
    interpolation='bicubic',
    seed=9876,
    batch_size=20,
    class_mode='categorical')

# show a grid with 20 training images drawn by the generator

plt.figure()
fig, axs = plt.subplots(4, 5, figsize=(10, 10), sharey=True, sharex=True)
train_generator.reset()
batch_images, batch_labels = train_generator.next()
print('First batch of labels:', len(batch_labels), '\n', np.argmax(batch_labels, axis=1))
for cell, sample in zip(axs.flatten('C'), batch_images.tolist()):
    cell.imshow(image.array_to_img(sample))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.RMSprop(lr=5e-4),
              metrics=[metrics.categorical_accuracy])

# perform the actual training
history = model.fit(
    train_generator,
    steps_per_epoch=5,
    epochs=90,)
model.save("test.h5")  # store the net with all trained weights for future use


# plot graphs that track performance of the neural net over all training epochs
import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



output_neurons = model.output
win = K.square(output_neurons[:, 2:3])

input_neurons = model.input
grads = K.gradients(win, input_neurons)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
outputs = [win, grads]
fetch_win_and_grads = K.function([input_neurons], outputs)


def gradient_ascent(x, iterations, step, max_win=None):
    for i in range(iterations):
        win_value, grad_values = fetch_win_and_grads([x])
        if max_win is not None and win_value > max_win:
            break
        if i % 5 == 0:
            print('Win at iteration', i, ':', win_value[0])
        x += step * grad_values
    return x


starting_point = np.full((100, 100, 3), 0.5)  # 0.5 is medium gray
plt.figure()
plt.imshow(starting_point)


feed_to_net = np.expand_dims(starting_point, axis=0)
result_from_net = gradient_ascent(feed_to_net,
        iterations=300,
        step=0.005)


ideal_img = np.clip(np.copy(result_from_net[0]), 0, 1.0)
plt.figure()
plt.imshow(ideal_img)
plt.show()
