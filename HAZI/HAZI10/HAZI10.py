import tensorflow as tf

def mnist_digit_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return train_images / 255.0, train_labels, test_images / 255.0, test_labels


def mnist_model():
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(10),
                                 tf.keras.layers.Softmax()])

    return model


def model_compile(model: tf.keras.Sequential):
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


def model_fit(model: tf.keras.Sequential, epochs, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=epochs)
    return model


def model_evaluate(model: tf.keras.Sequential, test_images, test_labels):
    return model.evaluate(test_images, test_labels)


# train_images, train_labels, test_images, test_labels = mnist_digit_data()
# model = mnist_model()
# model = model_compile(model)
# model = model_fit(model, 10, train_images, train_labels)
# test_loss, test_acc = model_evaluate(model, test_images,  test_labels)
# print('\nloss:', test_loss)
# print('\naccuracy:', test_acc)