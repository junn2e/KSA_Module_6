import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt

#그래프
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
#하이퍼파라미터 설정
np.random.seed(1337) #for reproducibility

batch_size = 256
nb_classes = 10
nb_epoch = 5

#input image dimensions
img_rows, img_cols = 28, 28
#number of convolutional filters to use
nb_filters = 32

#데이터 로드
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#tensorflow backend 사용시 (Height, Width, channel)순 입력, Theano backend 사용시 (channel, Height, Width)순 입력
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#데이터 정규화 부분
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train sample')
print(X_test.shape[0], 'test sample')

#정답 label one-hot encoding
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

input_tensor = Input(shape=input_shape)
x = Conv2D(nb_filters, kernel_size=(3,3), padding='valid', name='Conv1')(input_tensor)
# x = BatchNormalization()(x)
x = Activation('relu', name='relu_1')(x)
x = Conv2D(nb_filters, kernel_size=(3,3), padding='valid', name='Conv2')(x)
x = Activation('relu', name='relu_2')(x)
x = MaxPooling2D(pool_size=(2,2), name='pool_1')(x)
# x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(units=128, name='hidden_1')(x)
x = Activation('relu', name='relu_3')(x)
x = Dense(units=nb_classes, name='hidden_2')(x)

output_tensor = Activation('softmax', name='output_tensor')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_split=0.2)

print('Test start')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()