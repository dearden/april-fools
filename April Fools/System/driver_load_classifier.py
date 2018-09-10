import pickle
from classify import train_classify


clf_name = "standard_linear_svc.sav"

# open a file, where you stored the pickled data
file = open('clfs/{}'.format(clf_name), 'rb')
# dump information to that file
clf = pickle.load(file)

train_classify(['feats'], clf)
