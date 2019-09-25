import pickle

with open('logs/learning_curves.pickle', 'rb') as handle:
    objs = []
    while 1:
        try:
            objs.append(pickle.load(handle))
        except EOFError:
            break
    print(objs)