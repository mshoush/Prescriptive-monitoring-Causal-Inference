import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle

import xgboost as xgb

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample


def create_and_evaluate_model(args):
    global trial_nr
    if trial_nr % 50 == 0:
        print(trial_nr)
    trial_nr += 1
    
    score = 0
    for current_train_names, current_test_names in dataset_manager.get_idx_split_generator(dt_for_splitting, n_splits=n_splits):
        
        train_idxs = case_ids.isin(current_train_names)
        X_train = X_all[train_idxs]
        y_train = y_all[train_idxs]
        X_test = X_all[~train_idxs]
        y_test = y_all[~train_idxs]
        
        cls = xgb.XGBClassifier(objective='binary:logistic',
                          n_estimators=500,
                          learning_rate= args['learning_rate'],
                          subsample=args['subsample'],
                          max_depth=int(args['max_depth']),
                          colsample_bytree=args['colsample_bytree'],
                          min_child_weight=int(args['min_child_weight']),
                          seed=22,
                          n_jobs=-1)
        
        cls.fit(X_train, y_train)
        preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
        
        preds = cls.predict_proba(X_test)[:,preds_pos_label_idx]

        score += roc_auc_score(y_test, preds)
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


print('Preparing data...')
start = time.time()

dataset_name = argv[1] # prepared_bpic2017
params_dir = argv[2] #

train_ratio = 0.8
n_splits = 3

trial_nr = 1

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()

min_prefix_length = 1
if "prepared_bpic2017" in dataset_name:
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.95))
elif dataset_name == "uwv" or dataset_name == "bpic2018":
    max_prefix_length = dataset_manager.get_pos_case_length_quantile(data, 0.9)
else:
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.95))

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols, 
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                    'fillna': True}
    
# split into training and test
train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
# generate data where each prefix is a separate instance
dt_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)

# encode all prefixes
feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
X_all = feature_combiner.fit_transform(dt_prefixes)
y_all = np.array(dataset_manager.get_label_numeric(dt_prefixes))

# generate dataset that will enable easy splitting for CV - to guarantee that prefixes of the same case will remain in the same chunk
case_ids = dt_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]
dt_for_splitting = pd.DataFrame({dataset_manager.case_id_col: case_ids, dataset_manager.label_col: y_all}).drop_duplicates()

print('Optimizing parameters...')

space = {#'n_estimators': hp.choice('n_estimators', np.arange(150, 1000, dtype=int)),
         'learning_rate': hp.uniform("learning_rate", 0, 1),
         'subsample': hp.uniform("subsample", 0.5, 1),
         'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
         'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
         'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}

trials = Trials()
best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=10, trials=trials)

best_params = hyperopt.space_eval(space, best)

outfile = os.path.join(params_dir, "optimal_params_xgboost_%s.pickle" % (dataset_name))
# write to file
with open(outfile, "wb") as fout:
    pickle.dump(best_params, fout)
