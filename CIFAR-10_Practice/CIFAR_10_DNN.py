import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
# actual_single = CLASSES[y_train]
# plt.imshow(x_train[20], interpolation="bicubic")
# tmp = "Label:" + str(actual_single[20])
# plt.title(tmp, fontsize=30)
# plt.tight_layout()
# plt.show()




NUM_CLASSES = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#모델 만들기

input_layer = Input(shape=(32, 32, 3))

x = Flatten()(input_layer)
x = Dense(units=200, activation='relu')(x)
x = Dense(units=200, activation='relu')(x)
x = Dense(units=200)(x)
x = Activation('relu')(x)
x = Dense(units=150)(x)
x = Activation('relu')(x)

output_layer = Dense(units=10, activation='softmax')(x)

model = Model(input_layer, output_layer)
#모델 컴파일
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

#모델 훈련
history = model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()