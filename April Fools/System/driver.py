from classify import train_classify
from classify import group_classify
from classify import bayesian_optimisation_rf
from classify import bayesian_optimisation_svc
from classify import tune_svm_grid_search
from classify import create_classification_table
from classify import create_crossval_table
from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle


LINEAR_SVC = LinearSVC()
RBF_SVC = SVC()
FOREST = RandomForestClassifier()
TREE = DecisionTreeClassifier()
BAYES = MultinomialNB()
REGRESSION = LogisticRegression()

STD_SCAL = StandardScaler()
MMx_SCAL = MinMaxScaler(feature_range=(0, 1))
ROB_SCAL = RobustScaler(quantile_range=(25, 75))

create_crossval_table(LogisticRegression())

# # Do an untuned SVC
# train_classify(["fake_news_bag"], SVC(kernel='linear'))
#
# # group_classify(LINEAR_SVC)
#
# # train_classify(['sanity'], FOREST, ['complexity', 'details', 'deception'])
#
# Tune a random forest classifier
# params = bayesian_optimisation_rf(["feats"])
# TUNED_RF = RandomForestClassifier(max_features=params[0],
#                                   max_depth=params[1],
#                                   min_samples_split=params[2],
#                                   min_samples_leaf=params[3])
# train_classify(["feats"], TUNED_RF)
#
# # Maybe have a train method here. Makes sense to store a trained method.
#
# # save the model to disk
# filename = 'clfs/tuned_forest.sav'
# pickle.dump(TUNED_RF, open(filename, 'wb'))

# #################################################################################
# # Bayesian SVM Tuning
# #################################################################################
#
# params = bayesian_optimisation_svc(["feats"])
# TUNED_SVC = SVC(C=params[0],
#                 gamma=params[1],
#                 kernel='linear')
# train_classify(["feats"], TUNED_SVC)
#
# # save the model to disk
# filename = 'clfs/standard_linear_svc.sav'
# pickle.dump(TUNED_SVC, open(filename, 'wb'))
#
# #################################################################################


# # Tune an SVC with Grid Search
# GRID_SVC = tune_svm_grid_search(['feats'])
# train_classify(["feats"], GRID_SVC)
#
# # save the model to disk
# filename = 'clfs/grid_svc.sav'
# pickle.dump(GRID_SVC, open(filename, 'wb'))
