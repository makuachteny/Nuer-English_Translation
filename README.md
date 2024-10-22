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

import collections  # For data manipulation or handling collections
import load_func
import numpy as np
import tensorflow as tf

# Importing required functions and layers from Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
```

## Preprocessing

### Load the Data

Load the dataset containing English-Nuer sentence pairs.

```python

english_sentences = load_func.load_data('data/english.txt')
# Load French data
nuer_sentences = load_func.load_data('data/nuer.txt')

print('Dataset Loaded')
```

### Check loaded data

```python
for sample_i in range(2):
    print('english Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('nuer Line {}:  {}'.format(sample_i + 1, nuer_sentences[sample_i]))
```

### Tokenize the Data

Convert sentences into sequences of integers.

```python
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_t = Tokenizer()
    x_t.fit_on_texts(x)
    
    return x_t.texts_to_sequences(x), x_t

# Tokenize the sentences
text_tokenized, text_tokenizer = tokenize(english_sentences)

# Print the tokenized output
print()
for sample_i, (sent, token_sent) in enumerate(zip(english_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
print('  Output: {}'.format(token_sent))

```

### Pad the Sequences

Ensure all sequences have the same length.

```python
def padding(sequences, maxlen=None, padding='post', truncating='post', value=0):
    """
    Pad sequences to ensure they all have the same length.
    
    :param sequences: List of sequences (lists of integers).
    :param maxlen: Maximum length of the sequences. If None, it will be the length of the longest sequence.
    :param padding: 'pre' or 'post', where to add the padding.
    :param truncating: 'pre' or 'post', where to truncate sequences longer than maxlen.
    :param value: Value to use for padding.
    :return: Padded sequences as a 2D numpy array.
    """
    return pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

# Example usage
max_length = max(len(seq) for seq in text_tokenized)  # Determine the maximum length of the sequences
padded_sequences = padding(text_tokenized, maxlen=max_length)

# Print padded sequences
print("\nPadded Sequences:")
for sample_i, padded_sent in enumerate(padded_sequences):
    print(f'Sequence {sample_i + 1}: {padded_sent}')

```

### Model and Training

Train the model on the preprocessed data.

```python
model.fit(eng_sequences, np.array(nuer_sequences), epochs=10, batch_size=32, validation_split=0.2)
```


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
1. Performance: It has an average BLEU score that reflects its capability in generating coherent translations.
2. Challenges: The model struggles with longer sentences and complex grammar. More sophisticated techniques could improve the results.
3. Data: The quality and variety of the dataset are critical for better model performance.

## Potential Improvements

1. Data Augmentation: Expanding the dataset with more examples could improve performance.
2. Advanced Architectures: Try Transformer models for better translation quality.
3. Pre-trained Embeddings: Using pre-trained word embeddings like GloVe may improve the model's understanding of language.
4. Attention Mechanism: Implementing attention mechanisms can help the model focus on important parts of input sentences during translation.

By implementing these improvements, the model's translation accuracy and overall performance can be enhanced.
