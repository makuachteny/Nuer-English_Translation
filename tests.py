import unittest

# Sample tokenize function provided by you
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    
    # Initialize and fit tokenizer
    x_t = Tokenizer()
    x_t.fit_on_texts(x)
    
    # Tokenize the sentences
    return x_t.texts_to_sequences(x), x_t

class TestTokenizeFunction(unittest.TestCase):
    
    def setUp(self):
        self.text_sentences = [
            'The quick brown fox jumps over the lazy dog .',
            'By Jove , my quick study of lexicography won a prize .',
            'This is a short sentence .'
        ]

    def test_tokenize_output(self):
        # Run the tokenize function
        tokenized_output, tokenizer = tokenize(self.text_sentences)
        
        # Assert that the output is a tuple
        self.assertIsInstance(tokenized_output, list)
        self.assertIsInstance(tokenizer, object)
        
        # Assert the tokenizer's word index contains expected words
        expected_words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
                          'by', 'jove', 'my', 'study', 'of', 'lexicography', 'won', 'a',
                          'prize', 'this', 'is', 'short', 'sentence']
        for word in expected_words:
            self.assertIn(word, tokenizer.word_index)
        
        # Check if the tokenized sequences are the correct length
        self.assertEqual(len(tokenized_output), len(self.text_sentences))
        for tokenized_sentence in tokenized_output:
            self.assertGreater(len(tokenized_sentence), 0)
    
    def test_tokenizer_word_index(self):
        # Run the tokenize function
        _, tokenizer = tokenize(self.text_sentences)
        
        # Assert the tokenizer's word index size is greater than 0
        self.assertGreater(len(tokenizer.word_index), 0)
        
        # Check the tokenizer word index for a few specific words
        self.assertIn('quick', tokenizer.word_index)
        self.assertIn('dog', tokenizer.word_index)

if __name__ == '__main__':
    unittest.main()
