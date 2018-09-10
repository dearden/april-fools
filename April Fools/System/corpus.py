from article import Article
from get_features import get_features
import numpy as np


class Corpus(object):
    def __init__(self, articles):
        self.train_feats = []
        self.test_feats = []
        self.train_articles = []
        self.test_articles = []
        self.feat_groups = dict()
        self.feat_names = []

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

    def get_training_indices(self):
        indices = [article.index for article in self.train_articles]
        return indices

    def get_testing_indices(self):
        indices = [article.index for article in self.train_articles]
        return indices
