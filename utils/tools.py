import pickle
import os


def save_to_pickle(obj, filename):
    with open(f'save/{filename}', 'wb') as file:
        pickle.dump(obj, file)


def pickle_to_obj(filename):
    if os.path.exists(f'save/{filename}'):
        with open(f'save/{filename}', 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        return None


