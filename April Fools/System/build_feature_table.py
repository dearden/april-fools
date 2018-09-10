import os
import csv
from article import Article
from corpus_bags import CorpusBags
from corpus_feats import CorpusFeats
import numpy as np


META_PATH = "/home/ed/Documents/April Fools/Tagged_Corpus/meta.csv"
CORPUS_PATH = "/home/ed/Documents/April Fools/Tagged_Corpus/"


def build_feature_table(name, kind='feats', meta_path=META_PATH, corpus_path=CORPUS_PATH):
    print("Begin")
    # Open the meta file.
    with open(meta_path) as meta_file:
        reader = csv.reader(meta_file, delimiter=';')
        meta = [line for line in reader]
    print("Read in metadata")

    articles = []

    # Create article objects.
    print("Begin creating articles")
    for item in meta:
        article = Article(item[0], corpus_path + str(item[0]), item[5], item[2], item[6])
        articles.append(article)
        print("Read in Article %s\t--\t%s" % (article.index, article.head))
    print("All articles created")

    print("Begin creating corpus")
    if kind == 'feats':
        corpus = CorpusFeats(articles)
    elif kind == 'bag':
        corpus = CorpusBags(articles)
    print("Corpus created")

    train = corpus.get_training_table().tolist()
    train_labels = corpus.get_training_labels()
    train_indices = corpus.get_training_indices()
    test = corpus.get_testing_table().tolist()
    test_labels = corpus.get_testing_labels()
    test_indices = corpus.get_testing_indices()
    feature_names = corpus.get_feature_names()

    print("Commence Training Table Creation")
    # Add feature names and labels to training feature table
    train_table = [feature_names] + train
    train_table[0] = ['index'] + train_table[0] + ['class']
    train_table[1:] = [[index] + row + [label] for index, row, label in
                       zip(train_indices, train_table[1:], train_labels)]
    with open(name + "_train.csv", "w") as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_table)
    print("Training Table Complete")

    print("Commence Testing Table Creation")
    # Add feature names and labels to testing feature table
    test_table = [feature_names] + test
    test_table[0] = ['index'] + test_table[0] + ['class']
    train_table[1:] = [[index] + row + [label] for index, row, label in
                       zip(test_indices, test_table[1:], test_labels)]
    with open(name + "_test.csv", "w") as test_file:
        writer = csv.writer(test_file)
        writer.writerows(test_table)
    print("Testing Table Complete")
    print("Finished")

    return corpus


if __name__ == "__main__":
    fake_news_meta_path = "/home/ed/Documents/April Fools/Tagged_Fake_News/meta.csv"
    fake_news_corp_path = "/home/ed/Documents/April Fools/Tagged_Fake_News/"
    build_feature_table("feats")
    build_feature_table("bow", kind='bag')
    build_feature_table("fake_news_feats", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path)
    build_feature_table("fake_news_bow", meta_path=fake_news_meta_path, corpus_path=fake_news_corp_path, kind='bag')
