import pickle


def save_array(array, filename):
    open_file = open(filename, "wb")
    pickle.dump(array, open_file)
    open_file.close()
