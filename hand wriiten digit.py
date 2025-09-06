import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Visualize predictions using Seaborn theme
sns.set_theme()
num_images_to_display = 6  # Change this number based on your preference

plt.figure(figsize=(12, 12))

for i in range(num_images_to_display):
    plt.subplot(2, num_images_to_display // 2, i + 1)
    image_index = i  # Change this index based on your preference
    predict = x_test[image_index].reshape(28, 28)
    pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
    plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
    plt.title("Predicted Label: " + str(pred.argmax()))

plt.show()