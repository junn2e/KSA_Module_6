from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.regularizers import l2

def build_model_16(img_height, img_width, img_channel, class_count, weight_decay):
    input_layer = Input(shape=(img_height, img_width, img_channel))
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_1')(input_layer)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_3')(input_layer)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_5')(input_layer)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_6')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_7')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_8')(input_layer)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_9')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_10')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_11')(input_layer)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_12')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_13')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    output_layer = Dense(class_count, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    return model

model_build = build_model_16(224, 224, 3, 1000, 1e-4)
model_build.summary()