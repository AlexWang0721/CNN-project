# 实战2： CNN卷积神经网络—CIFAR10图像分类项目
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# 1. 加载和预处理数据集
# CIFAR10数据集包含60000张32×32彩色图像，分类10个类别
(train_images,train_labels),(test_images,test_labels) = datasets.cifar10.load_data()
# 将像素值归一化到 0-1之间（原始像素值范围是0-255）
train_images, test_images = train_images/255.0, test_images/255.0
# 将标签转换为 one-hot编码（可选，但推荐用于分类任务）
train_labels = tf.keras.utils.to_categorical(train_labels,10)
test_labels = tf.keras.utils.to_categorical(test_labels,10)
# 2, 构建卷积神经网络模型
model = models.Sequential([
    # 卷积层1:32个3×3滤波器，激活函数relu，输入形状32×32×3
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    # 卷积层2:64个3×3滤波器
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    # 卷积层3：64个3x3滤波器
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    # 全连接层
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # 丢弃层防止过拟合
    # 输出层： 10个神经元对应10个类别，使用softmax激活
    layers.Dense(10,activation='softmax')
])
# 3.编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 数据增强训练
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    horizontal_flip=True)
# 4. 训练模型
history = model.fit(datagen.flow(train_images,train_labels,batch_size=64),
                    epochs=50,
                    validation_data=(test_images,test_labels))
# 5.评估模型
test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
print(f'\nTest accuracy:{test_acc*100:.2f}%')
# 6.可视化训练过程
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# 7.进行预测
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 取测试集第一张图片
img = test_images[0]
prediction = model.predict(tf.expand_dims(img, axis=0))  # 添加批次维度
predicted_class = class_names[tf.argmax(prediction[0]).numpy()]
true_class = class_names[tf.argmax(test_labels[0]).numpy()]

print(f'\nPredicted: {predicted_class}, True: {true_class}')
