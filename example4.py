import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

data = [
    "This is a long text that needs summarization.",
    "Another example of a long text document that should be summarized."
]

# Токенізатор
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1

# Підготовка послідовностей
sequences = tokenizer.texts_to_sequences(data)
max_sequence_len = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len)

# Побудова моделі
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_len))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

# Компіляція моделі
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
# Потрібні вхідні дані (X) та мітки (y) для навчання
# model.fit(X, y, epochs=10)

# Генерація резюме
def summarize_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_len)
    summary = model.predict(padded_sequence)
    return summary
