import os

def load_data(path):
    """ Load data from a file """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data.split('\n')  # Split the data by new line