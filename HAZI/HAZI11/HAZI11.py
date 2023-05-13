import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def cifar100_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    test_images = test_images / 255.0
    train_images = train_images / 255.0

    return train_images, train_labels, test_images, test_labels


def cifar100_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(100, activation='softmax')
    ])

    return model


def model_compile(model):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def model_fit(model, epochs, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=epochs)
    return model


def model_evaluate(model, test_images, test_labels):
    return model.evaluate(test_images, test_labels, verbose=2)


train_images, train_labels, test_images, test_labels = cifar100_data()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

print(train_images.shape)

model = cifar100_model()
model = model_compile(model)
model = model_fit(model, 3, train_images, train_labels)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nloss:', test_loss)
print('\naccuracy:', test_acc)
