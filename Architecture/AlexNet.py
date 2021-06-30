from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.regularizers import l2


def build_model(img_height, img_width, img_channel, class_count, weight_decay):
    input_layer = Input(shape=(img_height, img_width, img_channel))
    x = Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_1')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(256, kernel_size=(5, 5), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_3')(x)
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_4')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='relu', name='Conv2D_5')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output_layer = Dense(class_count, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    return model


model_build = build_model(227, 227, 3, 1000, 1e-4)
model_build.summary()
