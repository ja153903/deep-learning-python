# Libraries to import
from keras.datasets import imdb
from keras import preprocessing
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential

# number of words to consider as features
max_features = 10000

# cuts off the text after this number of words
maxlen = 20

# load data as list of integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# turn list of integers into a 2D integer tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
