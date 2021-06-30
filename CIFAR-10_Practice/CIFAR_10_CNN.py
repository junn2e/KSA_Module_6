import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc = 0)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

save_dir = os.path.join(os.getcwd(), 'saved_model')
model_name = 'keras_cifar10_aug_trained_model_upgrade.h5' #.h5/.hdf5 -> keras 대용량 파일 저장과 배포

img_rows, img_cols = 32, 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
# actual_single = CLASSES[y_train]
# plt.imshow(x_train[20], interpolation="bicubic")
# tmp = "Label:" + str(actual_single[20])
# plt.title(tmp, fontsize=30)
# plt.tight_layout()
# plt.show()
#tensorflow backend 사용시 (Height, Width, channel)순 입력, Theano backend 사용시 (channel, Height, Width)순 입력
if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

NUM_CLASSES = 10

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#모델 만들기

input_layer = Input(shape=(32, 32, 3))

x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', strides=1, padding='same', name='Conv1')(input_layer)
x = LeakyReLU()(x)
x = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', strides=2, padding='same', name='Conv2')(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal',  strides=1, padding='same', name='Conv3')(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal', strides=2, padding='same', name='Conv4')(x)
x = LeakyReLU()(x)
x = Flatten()(x)
x = Dense(128, kernel_initializer='he_normal')(x)
x = LeakyReLU()(x)
x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)
model.summary()
#모델 컴파일
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# data aug
datagen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

#모델 학습 (aug 적용)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                              steps_per_epoch=x_train.shape[0]/16,
                              epochs=20,
                              validation_data=(x_test, y_test),
                              workers=4)

#모델 학습 (aug 미적용)
# history = model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=1, validation_split=0.2)
#

#모델 저장
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

#predict
print("Test start")
score = model.evaluate(x_test, y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()