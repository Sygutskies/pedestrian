import tensorflow as tf

class HarModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, activation='relu')
        self.ln1 = tf.keras.layers.LayerNormalization(axis=1)
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
        self.lstm3 = tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
        self.ln2 = tf.keras.layers.LayerNormalization(axis=1)
        self.lstm4 = tf.keras.layers.LSTM(64, return_sequences=True, activation='relu')
        self.lstm5 = tf.keras.layers.LSTM(64, return_sequences=True, activation='relu')
        self.ln3 = tf.keras.layers.LayerNormalization(axis=1)
        self.lstm6 = tf.keras.layers.LSTM(64, return_sequences=False, activation='relu')
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(32, activation='relu')
        self.out_1 = tf.keras.layers.Dense(3, activation='softmax')
        self.out_2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.ln1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.ln2(x)
        x = self.lstm4(x)
        x = self.lstm5(x)
        x = self.ln3(x)
        x = self.lstm6(x)
        x = self.d1(x)
        x = self.d2(x)
        return tf.keras.layers.concatenate([self.out_1(x), self.out_2(x)])