import tensorflow as tf


class CNN(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self._model_fn()

    def _model_fn(self):
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)

        self.cross_entropy_loss = -tf.reduce_sum(self.y * tf.log(x))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy_loss)

        correct_prediction = tf.equal(tf.argmax(x, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.summary.scalar('loss', self.cross_entropy_loss)
        tf.summary.scalar('acc', self.accuracy)
        self.summary_op = tf.summary.merge_all()
