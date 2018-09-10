from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics
import numpy as np

import scipy
import pickle

#BO libraries
import GPy
import GPyOpt
rs = 1234


sns.set(color_codes=True)

FEAT_TABLE_NAMES = ["feats"]
all_groups = ["complexity", "deception", "details", "formality", "humour", "imagination", "vagueness"]
groups = ["deception"]

#Classification gubbins
K_FOLDS = 10


def get_feature_table(names, train_test):
    tables = []
    for name in names:
        table = pd.read_csv("%s_%s.csv" % (name, train_test))
        table.set_index('index', inplace=True)
        labels = table['class']
        table.drop('class', axis=1, inplace=True)
        tables.append(table)
    feature_table = pd.concat(tables, axis=1, join='inner')
    return feature_table, labels


def get_feature_groups(groups, feature_table):
    if groups == ["all"]:
        return feature_table
    features = []
    for group in groups:
        with open("feature_sets/{}.txt".format(group)) as file:
            for line in file:
                feat = line.strip()
                if feat in feature_table.columns:
                    features.append(feat)
    return feature_table[features]


def train_classify(tables, classifier, groups=['all'], scaler=StandardScaler(), confusion=False):
    training_table, training_labels = get_feature_table(tables, "train")
    training_table = get_feature_groups(groups, training_table)

    pipeline = Pipeline([
        ('normalizer', scaler),  # Step1 - normalize data
        ('clf', classifier)  # Step2 - classifier
    ])

    predicted = cross_val_predict(pipeline, training_table, training_labels, cv=K_FOLDS)
    # predicted = [1 for x in range(len(training_labels))]

    score = metrics.f1_score(training_labels, predicted, pos_label=1)
    precision = metrics.average_precision_score(training_labels, predicted)
    recall = metrics.recall_score(training_labels, predicted)
    accuracy = metrics.accuracy_score(training_labels, predicted)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Accuracy: {}".format(accuracy))
    print("F1-Score: {}".format(score))

    feats_string = "_".join(FEAT_TABLE_NAMES)
    with open("{0}_predictions.txt".format(feats_string), 'w') as pred_file:
        for p in predicted:
            pred_file.write("{}\n".format(p))

    if confusion:
        print(confusion_matrix(training_labels, predicted))

    return predicted


def train_model(tables, classifier, groups=['all']):
    training_table, training_labels = get_feature_table(tables, "train")
    training_table = get_feature_groups(groups, training_table)

    pipeline = Pipeline([
        ('normalizer', StandardScaler()),  # Step1 - normalize data
        ('clf', classifier)  # Step2 - classifier
    ])

    classifier.fit(training_table, training_labels)
    return classifier


def test_model(tables, classifier, groups=['all'], confusion=False, printout=True):
    testing_table, testing_labels = get_feature_table(tables, "test")
    testing_table = get_feature_groups(groups, testing_table)

    scaled = StandardScaler().fit_transform(testing_table)
    predicted = classifier.predict(scaled)

    score = metrics.f1_score(testing_labels, predicted, pos_label=1)
    precision = metrics.average_precision_score(testing_labels, predicted)
    recall = metrics.recall_score(testing_labels, predicted)
    accuracy = metrics.accuracy_score(testing_labels, predicted)

    if printout:
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Accuracy: " + str(accuracy))
        print("F1-Score:", score)

    feats_string = "_".join(FEAT_TABLE_NAMES)
    with open("{0}_test_predictions.txt".format(feats_string), 'w') as pred_file:
        for p in predicted:
            pred_file.write("{}\n".format(p))

    if confusion:
        print(confusion_matrix(testing_labels, predicted))

    return predicted


def group_classify(classifier, table='feats'):
    training_table, training_labels = get_feature_table([table], "train")
    columns = ['precision', 'recall', 'accuracy', 'f1']
    results_df = pd.DataFrame(np.zeros((len(all_groups), len(columns))), index=all_groups, columns=columns)
    results_df.index.name = "group"
    temp_groups = all_groups + ['all']
    for group in temp_groups:
        print("\n{}".format(group))

        predicted = train_classify([table], classifier, [group])

        results = dict()
        results['precision'] = metrics.average_precision_score(training_labels, predicted)
        results['recall'] = metrics.recall_score(training_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
        results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

        results_df.at[group, :] = results.values()

    results_df.to_csv("group_results.csv")
    return results_df


# This method is not reusable in any way. Just creates the feature table for the paper.
def create_classification_table():
    feat_testing_table, testing_labels = get_feature_table(['feats'], "test")

    row_names = ['baseline'] + all_groups + ['feats', 'bow', 'bop', 'bos', 'bag', 'all']

    columns = ['precision', 'recall', 'accuracy', 'f1']
    results_df = pd.DataFrame(np.zeros((len(row_names), len(columns))), index=row_names, columns=columns)
    results_df.index.name = "Features"

    print("\nBaseline")

    predicted = np.ones(len(testing_labels))

    results = dict()
    results['precision'] = metrics.average_precision_score(testing_labels, predicted)
    results['recall'] = metrics.recall_score(testing_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(testing_labels, predicted)
    results['f1'] = metrics.f1_score(testing_labels, predicted, pos_label=1)

    results_df.at["baseline", :] = results.values()

    for group in all_groups:
        print("\n{}".format(group))

        clf = train_model(['feats'], LinearSVC(), [group])
        predicted = test_model(['feats'], clf, [group])

        results = dict()
        results['precision'] = metrics.average_precision_score(testing_labels, predicted)
        results['recall'] = metrics.recall_score(testing_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(testing_labels, predicted)
        results['f1'] = metrics.f1_score(testing_labels, predicted, pos_label=1)

        results_df.at[group, :] = results.values()

    for table in ['feats', 'bow', 'bop', 'bos']:
        print("\n{}".format(table))

        clf = train_model([table], LinearSVC())
        predicted = test_model([table], clf)

        results = dict()
        results['precision'] = metrics.average_precision_score(testing_labels, predicted)
        results['recall'] = metrics.recall_score(testing_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(testing_labels, predicted)
        results['f1'] = metrics.f1_score(testing_labels, predicted, pos_label=1)

        results_df.at[table, :] = results.values()

    print("\nbag")

    clf = train_model(['bow', 'bop', 'bos'], LinearSVC())
    predicted = test_model(['bow', 'bop', 'bos'], clf)

    results = dict()
    results['precision'] = metrics.average_precision_score(testing_labels, predicted)
    results['recall'] = metrics.recall_score(testing_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(testing_labels, predicted)
    results['f1'] = metrics.f1_score(testing_labels, predicted, pos_label=1)

    results_df.at['bag', :] = results.values()

    print("\nall")

    clf = train_model(['feats', 'bow', 'bop', 'bos'], LinearSVC())
    predicted = test_model(['feats', 'bow', 'bop', 'bos'], clf)

    results = dict()
    results['precision'] = metrics.average_precision_score(testing_labels, predicted)
    results['recall'] = metrics.recall_score(testing_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(testing_labels, predicted)
    results['f1'] = metrics.f1_score(testing_labels, predicted, pos_label=1)

    results_df.at['all', :] = results.values()

    return results_df


# This method is not reusable in any way. Just creates the feature table for the paper.
def create_crossval_table(classifier=LinearSVC()):
    feat_training_table, training_labels = get_feature_table(['feats'], "train")

    row_names = all_groups + ['feats', 'comp + det', 'selection', 'top_6', 'bow', 'bow + feats', 'all']

    columns = ['precision', 'recall', 'accuracy', 'f1']
    results_df = pd.DataFrame(np.zeros((len(row_names), len(columns))), index=row_names, columns=columns)
    results_df.index.name = "Features"

    for group in all_groups:
        print("\n{}".format(group))

        predicted = train_classify(['feats'], classifier, [group])

        results = dict()
        results['precision'] = metrics.average_precision_score(training_labels, predicted)
        results['recall'] = metrics.recall_score(training_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
        results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

        results_df.at[group, :] = results.values()

    print("\nComplexity + Detail")
    # Complexity + Detail
    predicted = train_classify(['feats'], classifier, groups=['complexity', 'details'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['comp + det', :] = results.values()

    # Selection
    print("\nSelection")
    predicted = train_classify(['feats'], classifier, groups=['fs_t5'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['selection', :] = results.values()

    # Top 6
    print("\nTop 6")
    predicted = train_classify(['feats'], classifier, groups=['t6'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['top_6', :] = results.values()

    for table in ['feats', 'bow']:
        print("\n{}".format(table))

        predicted = train_classify([table], classifier)

        results = dict()
        results['precision'] = metrics.average_precision_score(training_labels, predicted)
        results['recall'] = metrics.recall_score(training_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
        results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

        results_df.at[table, :] = results.values()

    print("\nbow + feats")

    predicted = train_classify(['bow', 'feats'], classifier)

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['bow + feats', :] = results.values()

    print("\nall")

    predicted = train_classify(['feats', 'bow'], classifier)

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['all', :] = results.values()

    return results_df


# This method is not reusable in any way. Just creates the feature table for the paper.
def create_fake_crossval_table(classifier=LinearSVC()):
    feat_training_table, training_labels = get_feature_table(['fake_news_feats'], "train")

    row_names = all_groups + ['fake_news_feats', 'comp + det', 'selection', 'top_6', 'fake_news_bow', 'bow + feats']

    columns = ['precision', 'recall', 'accuracy', 'f1']
    results_df = pd.DataFrame(np.zeros((len(row_names), len(columns))), index=row_names, columns=columns)
    results_df.index.name = "Features"

    for group in all_groups:
        print("\n{}".format(group))

        predicted = train_classify(['fake_news_feats'], classifier, [group])

        results = dict()
        results['precision'] = metrics.average_precision_score(training_labels, predicted)
        results['recall'] = metrics.recall_score(training_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
        results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

        results_df.at[group, :] = results.values()

    print("\nComplexity + Detail")
    # Complexity + Detail
    predicted = train_classify(['fake_news_feats'], classifier, groups=['complexity', 'details'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['comp + det', :] = results.values()

    # Selection
    print("\nSelection")
    predicted = train_classify(['fake_news_feats'], classifier, groups=['fs_t5'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['selection', :] = results.values()

    # Selection
    print("\nTop 6")
    predicted = train_classify(['fake_news_feats'], classifier, groups=['t6'])

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['top_6', :] = results.values()

    for table in ['fake_news_feats', 'fake_news_bow']:
        print("\n{}".format(table))

        predicted = train_classify([table], classifier)

        results = dict()
        results['precision'] = metrics.average_precision_score(training_labels, predicted)
        results['recall'] = metrics.recall_score(training_labels, predicted)
        results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
        results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

        results_df.at[table, :] = results.values()

    print("\nbow + feats")

    predicted = train_classify(['fake_news_bow', 'fake_news_feats'], classifier)

    results = dict()
    results['precision'] = metrics.average_precision_score(training_labels, predicted)
    results['recall'] = metrics.recall_score(training_labels, predicted)
    results['accuracy'] = metrics.accuracy_score(training_labels, predicted)
    results['f1'] = metrics.f1_score(training_labels, predicted, pos_label=1)

    results_df.at['bow + feats', :] = results.values()

    return results_df


def tune_svm_grid_search(tables, groups=['all']):
    training_table, training_labels = get_feature_table(tables, "train")
    training_table = get_feature_groups(groups, training_table)

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['linear', 'rbf']
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}
    grid_search = GridSearchCV(SVC(), param_grid, cv=K_FOLDS, scoring=make_scorer(precision_score))
    grid_search.fit(training_table, training_labels)
    return grid_search.best_estimator_


X_train, Y_train = get_feature_table(["feats"], "train")

# Courtesy of Henry Moss
# set up the function to optimize: The performance of our ML model
def fit_RF(x):
    x = np.atleast_2d(x)
    fs = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        clf=RandomForestClassifier(random_state=1234,n_estimators=1000,max_features=x[i,0],max_depth=int(x[i,1]),min_samples_split=x[i,2],min_samples_leaf=x[i,3],n_jobs=2)
        pipeline = Pipeline([
            # ('normalizer', MinMaxScaler(feature_range=(0, 1))),           # Step1 - normalize data
            # ('normalizer', RobustScaler(quantile_range=(25, 75))),        # Step1 - normalize data
            ('normalizer', StandardScaler()),  # Step1 - normalize data
            ('clf', clf)  # Step2 - classifier                                       # Step2 - classifier
        ])
        fs[i]=-np.mean(cross_validate(pipeline, X_train, Y_train, cv=5,n_jobs=2)['test_score'])
    return fs


def fit_SVC(x):
    x = np.atleast_2d(x)
    fs = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        clf = SVC(C=x[i,0], gamma=x[i,1], kernel='linear')
        pipeline = Pipeline([
            # ('normalizer', MinMaxScaler(feature_range=(0, 1))),           # Step1 - normalize data
            # ('normalizer', RobustScaler(quantile_range=(25, 75))),        # Step1 - normalize data
            ('normalizer', StandardScaler()),                              # Step1 - normalize data
            ('clf', clf)  # Step2 - classifier                                       # Step2 - classifier
        ])
        fs[i] = -np.mean(cross_validate(pipeline, X_train, Y_train, cv=5, n_jobs=2)['test_score'])
    return fs


def bayesian_optimisation_rf(tables, groups=['all']):
    training_table, training_labels = get_feature_table(tables, "train")
    training_table = get_feature_groups(groups, training_table)

    # set up domain of parameters
    # specify if continous and give a range
    # or if discrete give a list of possible values
    domain = [{'name': 'max_features', 'type': 'continuous', 'domain': (0.00001, 1)},
              {'name': 'max_depth', 'type': 'discrete', 'domain': tuple(range(2, 6))},
              {'name': 'min_samples_split', 'type': 'continuous', 'domain': (0.00000001, 0.5)},
              {'name': 'min_samples_leaf', 'type': 'continuous', 'domain': (0.00000001, 0.5)}]

    # set up and initialize BO model
    opt = GPyOpt.methods.BayesianOptimization(f=fit_RF,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='LCB',
                                              initial_design_type="random",
                                              initial_design_numdata=15,
                                              kernel=GPy.kern.Matern52(len(domain))
                                              # type of Gaussian Process most appropriate for ML models
                                              )

    # perform max_iter evaluations
    opt.run_optimization(max_iter=66)

    # # return score of successively evaluated points
    # scores = opt.Y * [-1]
    # # and plot the best previously observed solution at each iteration
    # plt.plot([x for x in range(1, len(scores))], [np.max(scores[:i]) for i in range(1, len(scores))])
    # plt.xlabel("Number of Evaluations")
    # plt.ylabel("Accuracy")
    # plt.axhline(y=.68, linewidth=2, color='red')
    # plt.show()
    return opt.x_opt


def bayesian_optimisation_svc(tables, groups=['all']):
    training_table, training_labels = get_feature_table(tables, "train")
    training_table = get_feature_groups(groups, training_table)

    # set up domain of parameters
    # specify if continous and give a range
    # or if discrete give a list of possible values
    domain = [{'name': 'C', 'type': 'continuous', 'domain': (0.001, 10.)},
              {'name': 'gamma', 'type': 'continuous', 'domain': (0.001, 1.)}]

    # set up and initialize BO model
    opt = GPyOpt.methods.BayesianOptimization(f=fit_SVC,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='LCB',
                                              initial_design_type="random",
                                              initial_design_numdata=15,
                                              kernel=GPy.kern.Matern52(len(domain))
                                              # type of Gaussian Process most appropriate for ML models
                                              )

    # perform max_iter evaluations
    opt.run_optimization(max_iter=66)

    # return score of successively evaluated points
    scores = opt.Y * [-1]
    # and plot the best previously observed solution at each iteration
    plt.plot([x for x in range(1, len(scores))], [np.max(scores[:i]) for i in range(1, len(scores))])
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Accuracy")
    plt.axhline(y=.68, linewidth=2, color='red')
    plt.show()
    return opt.x_opt
