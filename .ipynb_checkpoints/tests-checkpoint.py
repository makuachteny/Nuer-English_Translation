# tests.py

def test_tokenize(tokenize_function):
    """
    Test function for the tokenize function.
    """
    test_input = [
        'The quick brown fox jumps over the lazy dog .',
        'By Jove , my quick study of lexicography won a prize .',
        'This is a short sentence .'
    ]
    
    expected_output = [
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [10, 11, 1, 12, 13, 14, 15, 16, 17, 18, 1],
        [19, 20, 1, 21, 22, 1]
    ]
    
    # Get the tokenized output
    tokenized_output, tokenizer = tokenize_function(test_input)
    
    # Check if the lengths match
    assert len(tokenized_output) == len(expected_output), \
        f"Expected {len(expected_output)} sequences, but got {len(tokenized_output)}."
    
    # Check the actual tokenized sequences
    for i, (output, expected) in enumerate(zip(tokenized_output, expected_output)):
        assert output == expected, f"Test case {i + 1} failed. Expected {expected}, but got {output}."
    
    print("All tests passed.")
