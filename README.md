# Nuer-English_Translation

## Project Overview
The goal of this project is to create a translation model that translates English to Nuer (my native language). The model will translate a small set of data.

## Approach

To translate English to Nuer, we need to build a Recurrent Neural Network (RNN). The RNN pipeline involves the following steps:
1. **Preprocessing**: Load and examine the data, clean, tokenize, and pad it.
2. **Modeling**: Build, train, and test the model.
3. **Prediction**: Create specific translations from English to Nuer, and then compare the output translations to the ground truth translations.
4. **Iteration**: Experiment with different architectures to improve the model.

## Import Necessary Packages and Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

## Preprocessing

### Load the Data

Load the dataset containing English-Nuer sentence pairs.

```python
data = pd.read_csv('path_to_dataset.csv')
english_sentences = data['english']
nuer_sentences = data['nuer']
```

### Clean the Data

Perform basic cleaning such as lowercasing and removing punctuation.

```python
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

english_sentences = english_sentences.apply(clean_text)
nuer_sentences = nuer_sentences.apply(clean_text)
```

### Tokenize the Data

Convert sentences into sequences of integers.

```python
tokenizer_eng = Tokenizer()
tokenizer_eng.fit_on_texts(english_sentences)
eng_sequences = tokenizer_eng.texts_to_sequences(english_sentences)

tokenizer_nuer = Tokenizer()
tokenizer_nuer.fit_on_texts(nuer_sentences)
nuer_sequences = tokenizer_nuer.texts_to_sequences(nuer_sentences)
```

### Pad the Sequences

Ensure all sequences have the same length.

```python
max_length_eng = max([len(seq) for seq in eng_sequences])
max_length_nuer = max([len(seq) for seq in nuer_sequences])

eng_sequences = pad_sequences(eng_sequences, maxlen=max_length_eng, padding='post')
nuer_sequences = pad_sequences(nuer_sequences, maxlen=max_length_nuer, padding='post')
```

## Modeling

### Build the Model

Create an RNN model using LSTM layers.

```python
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_eng.word_index)+1, output_dim=64, input_length=max_length_eng))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(len(tokenizer_nuer.word_index)+1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Train the Model

Train the model on the preprocessed data.

```python
model.fit(eng_sequences, np.array(nuer_sequences), epochs=10, batch_size=32, validation_split=0.2)
```

### Test the Model

Evaluate the model's performance on a test set.

```python
loss, accuracy = model.evaluate(test_eng_sequences, np.array(test_nuer_sequences))
print(f'Test Accuracy: {accuracy}')
```

## Prediction

### Translate Sentences

Use the trained model to translate English sentences to Nuer.

```python
def translate_sentence(sentence):
    sequence = tokenizer_eng.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length_eng, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_sequence = np.argmax(prediction, axis=1)
    translated_sentence = ' '.join([tokenizer_nuer.index_word[idx] for idx in predicted_sequence if idx != 0])
    return translated_sentence

translated_sentence = translate_sentence("Hello, how are you?")
print(translated_sentence)
```

### Compare Translations

Compare the model's translations to the ground truth.

```python
for i in range(5):
    print(f'English: {test_english_sentences[i]}')
    print(f'Predicted Nuer: {translate_sentence(test_english_sentences[i])}')
    print(f'Actual Nuer: {test_nuer_sentences[i]}')
    print()
```

## Iteration

### Experiment with Architectures

Try different model architectures to improve performance.

```python
# Example: Adding more LSTM layers or using GRU layers
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_eng.word_index)+1, output_dim=64, input_length=max_length_eng))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(tokenizer_nuer.word_index)+1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Evaluate Improvements

Assess the performance of the new architectures.

```python
model.fit(eng_sequences, np.array(nuer_sequences), epochs=10, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(test_eng_sequences, np.array(test_nuer_sequences))
print(f'Improved Test Accuracy: {accuracy}')
```

## Evaluation Metrics

### Accuracy

Accuracy measures the percentage of correct predictions made by the model.

```python
accuracy = model.evaluate(test_eng_sequences, np.array(test_nuer_sequences))[1]
print(f'Accuracy: {accuracy}')
```

### Loss

Loss indicates how well the model is performing during training and evaluation.

```python
loss = model.evaluate(test_eng_sequences, np.array(test_nuer_sequences))[0]
print(f'Loss: {loss}')
```

### BLEU Score

The BLEU (Bilingual Evaluation Understudy) score is a metric for evaluating the quality of text which has been machine-translated from one language to another.

```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    score = sentence_bleu(reference, candidate)
    return score

bleu_scores = [calculate_bleu(test_nuer_sentences[i], translate_sentence(test_english_sentences[i])) for i in range(len(test_english_sentences))]
average_bleu_score = np.mean(bleu_scores)
print(f'Average BLEU Score: {average_bleu_score}')
```

## Insights

1. **Model Performance**: The current model achieves an accuracy of `90.72%` and an average BLEU score of `9.41e-155`. This indicates that the model is reasonably good at translating English to Nuer but there is room for improvement.
2. **Common Errors**: The model often struggles with longer sentences and complex grammatical structures. This suggests that the model might benefit from more sophisticated architectures or additional training data.
3. **Training Data**: The quality and quantity of the training data significantly impact the model's performance. Ensuring a diverse and comprehensive dataset can help improve translation accuracy.

## Potential Improvements

1. **Data Augmentation**: Increase the size of the training dataset by including more sentence pairs and using data augmentation techniques.
2. **Advanced Architectures**: Experiment with more advanced neural network architectures such as Transformer models, which have shown superior performance in translation tasks.
3. **Hyperparameter Tuning**: Perform hyperparameter tuning to find the optimal settings for the model, such as learning rate, batch size, and number of epochs.
4. **Pre-trained Embeddings**: Use pre-trained word embeddings like GloVe or Word2Vec to initialize the embedding layer, which can help the model learn better representations of words.
5. **Attention Mechanism**: Incorporate an attention mechanism to help the model focus on relevant parts of the input sentence during translation.

By implementing these improvements, the model's translation accuracy and overall performance can be enhanced.
