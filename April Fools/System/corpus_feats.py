from corpus import Corpus
from get_features import get_features
import numpy as np


class CorpusFeats(Corpus):
    def __init__(self, articles):
        Corpus.__init__(self, articles)
        for article in articles:
            features = get_features(article)
            if article.train:                   # put feature dict in either testing or training.
                self.train_feats.append(features)
                self.train_articles.append(article)  # keep a list of all articles in the training set.
            else:
                self.test_feats.append(features)
                self.test_articles.append(article)  # keep a list of all articles in the testing set.

        self.feat_names = features.keys()

    def get_training_table(self):
        table = []
        for row in self.train_feats:
            new_row = []
            for feat in self.feat_names:
                new_row.append(row[feat])
            table.append(new_row)
        table = np.array(table)
        return table

    def get_testing_table(self):
        table = []
        for row in self.test_feats:
            new_row = []
            for feat in self.feat_names:
                new_row.append(row[feat])
            table.append(new_row)
        table = np.array(table)
        return table

    def get_feature_names(self):
        return list(self.feat_names)

    def get_training_labels(self):
        labels = [article.label for article in self.train_articles]
        return labels

    def get_testing_labels(self):
        labels = [article.label for article in self.test_articles]
        return labels
