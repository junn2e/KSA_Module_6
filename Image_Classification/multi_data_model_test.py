import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os


# 이미지 경로를 가져오는 함수
def search(dirname):
    img_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                img_list.append(path+'/'+filename)
    return img_list


# 경로에서 가져온 이미지를 배열(list)형태로 입력
def load_image(filename):
    img = load_img(filename, target_size=(200,200))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255

    return img

img_list = search('./test_img')

# 데이터 로드 및 acc 하는 부분

model = load_model('saved_model/8mul_model.h5')
model.summary()

# create data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_it = train_datagen.flow_from_directory('8_multi_data/train/',
                                       class_mode='categorical', batch_size=64, target_size=(200, 200))
val_it = test_datagen.flow_from_directory('8_multi_data/val/',
                                     class_mode='categorical', batch_size=64, target_size=(200, 200))
test_it = test_datagen.flow_from_directory('8_multi_data/test/',
                                     class_mode='categorical', batch_size=64, target_size=(200, 200))

_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
print('>accuracy : %.3f' % (acc * 100.0))

# 모델을 이용해서 그림그리기
classes = np.array(['dog', 'cat', 'car', 'airplane', 'person', 'flower', 'fruit', 'motorbike'])
for i in img_list:
    img = load_image(i)
    result = model.predict(img)
    preds_value = classes[np.argmax(result, axis=-1)]

    temp = 'predict' + str(preds_value)
    model_acc = 'model_acc' + str(acc*100.0)
    plt.figure(figsize=(10,10))
    plt.title(temp)
    plt.xlabel(model_acc, fontsize=20)
    plt.title(temp)
    plt.imshow(img[0])
    plt.show()