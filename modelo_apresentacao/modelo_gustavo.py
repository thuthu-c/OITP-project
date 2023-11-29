import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Exemplo de dados
apresentacoes = ["A apresentação foi informativa e clara.", "A estrutura da apresentação precisa de melhorias."]
feedbacks = [1, 0]  # 1 para feedback positivo, 0 para feedback negativo


# Remover stopwords e tokenizar as palavras
try:
    # Tenta carregar as stopwords
    stop_words = set(stopwords.words("portuguese"))
except LookupError:
    # Se não puder carregar, faz o download e tenta novamente
    nltk.download('stopwords')
    stop_words = set(stopwords.words("portuguese"))

def preprocess_text(text):
    try:
        # Tenta carregar o tokenizer
        words = word_tokenize("Test sentence")
    except LookupError:
        # Se não puder carregar, faz o download e tenta novamente
        nltk.download('punkt')
        words = word_tokenize("Test sentence")

    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return " ".join(filtered_words)

apresentacoes = [preprocess_text(apresentacao) for apresentacao in apresentacoes]


# Tokenização e padronização dos dados
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(apresentacoes)

total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(apresentacoes)
padded_sequences = pad_sequences(sequences)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 16, input_length=len(padded_sequences[0])),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(padded_sequences, feedbacks, epochs=10, verbose=2)
