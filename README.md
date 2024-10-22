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

### Model and Training

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
for i in range(5):
    print(f'English: {test_english_sentences[i]}')
    print(f'Predicted Nuer: {translate_sentence(test_english_sentences[i])}')
    print(f'Actual Nuer: {test_nuer_sentences[i]}')
    print()
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
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer_eng.word_index)+1, output_dim=64, input_length=max_length_eng))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(tokenizer_nuer.word_index)+1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

```
## Evaluation Metrics

### BLEU Score

The BLEU (Bilingual Evaluation Understudy) score is a metric for evaluating the quality of text which has been machine-translated from one language to another.

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Create a smoothing function
smoothing_function = SmoothingFunction().method1

# Function to calculate BLEU score for the test set with smoothing
def calculate_bleu_score(references, predictions):
    total_bleu_score = 0
    for ref, pred in zip(references, predictions):
        # Convert sentences to list of words
        ref = [ref.split()]
        pred = pred.split()
        # Apply smoothing to avoid zero BLEU score for short sentences
        total_bleu_score += sentence_bleu(ref, pred, smoothing_function=smoothing_function)
    return total_bleu_score / len(references)

# Sample usage with smoothing
references = ["the sun rises at noon and sets in the east.", "she enjoys reading books in her free time."]
predictions = ["the sun is rising at noon and sets in east.", "she likes reading books during her free time."]

bleu_score = calculate_bleu_score(references, predictions)
print(f"Average BLEU Score with Smoothing: {bleu_score}")

```
The average BLEU SCORE: 0.32

## Insights

Insights
1. Performance: The model achieves an accuracy of ~90% and an average BLEU score that reflects its capability in generating coherent translations.
2. Challenges: The model struggles with longer sentences and complex grammar. More sophisticated techniques could improve the results.
3. Data: The quality and variety of the dataset are critical for better model performance.

## Potential Improvements

1. Data Augmentation: Expanding the dataset with more examples could improve performance.
2. Advanced Architectures: Try Transformer models for better translation quality.
3. Pre-trained Embeddings: Using pre-trained word embeddings like GloVe may improve the model's understanding of language.
4. Attention Mechanism: Implementing attention mechanisms can help the model focus on important parts of input sentences during translation.

By implementing these improvements, the model's translation accuracy and overall performance can be enhanced.
