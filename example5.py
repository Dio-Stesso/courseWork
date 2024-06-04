import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# Приклад великого набору текстових даних
data = [
    "The movie was fantastic and very engaging.",
    "I did not like the film, it was boring and slow.",
    "This is a great product, highly recommend it.",
    "The service was terrible, will not return."
]

# Ініціалізація токенізатора
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1

# Підготовка послідовностей
sequences = tokenizer.texts_to_sequences(data)
max_sequence_len = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len)

# Вибір вхідних та вихідних даних
labels = [1, 0, 1, 0]  # 1 - позитивний відгук, 0 - негативний відгук
X = padded_sequences
y = tf.keras.utils.to_categorical(labels)

# Побудова моделі
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 класи (позитивний, негативний)

# Компіляція моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
model.fit(X, y, epochs=10, verbose=1)

# Класифікація нового тексту
def classify_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_len)
    prediction = model.predict(padded_sequence)
    return prediction

# Приклад класифікації
new_text = "The product was excellent and delivery was fast."
print(classify_text(new_text))
