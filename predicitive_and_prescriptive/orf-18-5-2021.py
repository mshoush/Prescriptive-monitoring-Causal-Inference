import pandas as pd

pd.set_option('display.max_columns', None)
import itertools
import econml
import warnings
warnings.filterwarnings('ignore')

# Main imports
from econml.ortho_forest import DMLOrthoForest, DROrthoForest
#from econml.causal_tree import CausalTree
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV

from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np
import pickle5 as pickle


from sklearn.pipeline import FeatureUnion

import time
import os
from sys import argv
import pickle

import xgboost as xgb





# creat XGBoost model
def create_model(param,  X_train, y_train, n_iter=100,):
    param['metric'] = ['auc', 'binary_logloss']
    param['objective'] = 'binary:logistic'
    param['verbosity'] = 0#-1
    train_data = xgb.DMatrix(X_train, label=y_train)
    xgbm = xgb.train(param, train_data, n_iter)
    return xgbm

dataset_name = argv[1]  # prepared_bpic2017
optimal_params_filename = argv[2]  # params_dir
results_dir = argv[3]  # results_dir

calibrate = False
split_type = "temporal"
oversample = False
calibration_method = "beta"

train_ratio = 0.8
val_ratio = 0.2

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

print('Preparing data...')
start = time.time()

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()

min_prefix_length = 1
max_prefix_length = int(np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9)))

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True}

# split into training and test
if split_type == "temporal":
    train, test = dataset_manager.split_data_strict(data, train_ratio, split=split_type)
else:
    train, test = dataset_manager.split_data(data, train_ratio, split=split_type)

train, val = dataset_manager.split_val(train, val_ratio)



# generate data where each prefix is a separate instance
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_train_prefixes.to_pickle(os.path.join(results_dir, "dt_train_prefixes.pkl"))

dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
dt_test_prefixes.to_pickle(os.path.join(results_dir, "dt_test_prefixes.pkl"))


dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
dt_val_prefixes.to_pickle(os.path.join(results_dir, "dt_val_prefixes.pkl"))

import gc
# encode all prefixes
feature_combiner = FeatureUnion(
    [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
# features
print("\n=================Start  Encoding================\n")
X_train = feature_combiner.fit_transform(dt_train_prefixes)
y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
t_train = dataset_manager.get_treatment_numeric(dt_train_prefixes)
train_encoded = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(t_train),], axis=1)# pd.DataFrame(X_val)])
#del dt_train_prefixes
del X_train
del y_train
del t_train
gc.collect()

X_test = feature_combiner.fit_transform(dt_test_prefixes)
y_test = dataset_manager.get_label_numeric(dt_test_prefixes)
t_test = dataset_manager.get_treatment_numeric(dt_test_prefixes)
test_encoded = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test), pd.DataFrame(t_test)], axis=1)# pd.DataFrame(X_val)])
#del dt_test_prefixes
del X_test
del y_test
del t_test
gc.collect()
#del dt_test_prefixes

#dt_val_prefixes = pd.read_csv('./encoded_data/dt_val_prefixes.csv', sep=';')
X_val = feature_combiner.fit_transform(dt_val_prefixes)
y_val = dataset_manager.get_label_numeric(dt_val_prefixes)
t_val = dataset_manager.get_treatment_numeric(dt_val_prefixes)
val_encoded = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val), pd.DataFrame(t_val)], axis=1)# pd.DataFrame(X_val)])
#del dt_val_prefixes
del X_val
del y_val
del t_val
import gc
gc.collect()
print("\n=================read static & agg Encoding data================\n")
# to get columns for df after encoding
# agg encoding
with open("dt_transformed_agg.pkl", "rb") as fh:
    df_agg = pickle.load(fh)

with open("dt_transformed_static.pkl", "rb") as fh:
    df_static = pickle.load(fh)

#
# df_agg = pd.read_csv("dt_transformed_agg.csv", sep=';')
# # static encoding
# df_static = pd.read_csv("dt_transformed_static.csv", sep=';')
# # encoded data with col names from 0 to n
static_agg_df = pd.concat([df_static, df_agg], axis=1)
#df = pd.concat([X_encoded, y_encoded, t_encoded, ts_data], axis=1)
train_encoded.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
test_encoded.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
val_encoded.columns = list(static_agg_df.columns) + ["Outcome"] + ["Treatment"]
print("\n train_encoded.columns")
print(train_encoded.columns)

train_data = train_encoded
#del train_encoded
valid_data = val_encoded
#del val_encoded
test_data = test_encoded
#del test_encoded



y_train = train_data['Outcome']
#del train_data['Outcome']
#del train_data['Treatment']

X_train = train_data.drop(['Outcome', "Treatment"], axis=1)

y_valid = valid_data['Outcome']
#del valid_data['Outcome']
#del valid_data['Treatment']
X_valid = valid_data.drop(['Outcome', "Treatment"], axis=1)


y_test = test_data['Outcome']
#del test_data['Outcome']
#del test_data['Treatment']
X_test = test_data.drop(['Outcome', "Treatment"], axis=1)


#print(X_train.shape)
#print(len(y_train))

#print(X_test.shape)
#print(len(y_test))


#print(X_valid.shape)
#print(len(y_valid))



# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

# train the model with pre-tuned parameters
with open(optimal_params_filename, "rb") as fin:
    best_params = pickle.load(fin)


print("Create modle...")
xgbm = create_model(best_params, X_train, y_train)


X_train = xgb.DMatrix(X_train)
X_test = xgb.DMatrix(X_test)
X_valid = xgb.DMatrix(X_valid)

print("Predict train...")
preds_train = xgbm.predict(X_train)  # predictions for train data
print("Predict test...")
preds_test = xgbm.predict(X_test)  # predictions for test data
print("Predict valid...")
preds_valid = xgbm.predict(X_valid)  # predictions for test data
#import pandas as pd

print("Save results")
dt_preds_train = pd.DataFrame({"predicted_proba": preds_train, "actual": y_train,})# "prefix_nr": dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"], "case_id": dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})
dt_preds_train.to_pickle(os.path.join(results_dir, "preds_train_%s.pkl" % dataset_name))
#dt_preds_train.to_csv.to_pickle("./predictive_and_prescriptive/prepared_bpic2017.pkl")


# write test set predictions
dt_preds_test = pd.DataFrame({"predicted_proba": preds_test, "actual": y_test,})#"prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"], "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})
dt_preds_test.to_pickle(os.path.join(results_dir, "preds_test_%s.pkl" % dataset_name))

# write test set predictions
dt_preds_valid = pd.DataFrame({"predicted_proba": preds_valid, "actual": y_valid,})#"prefix_nr": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"], "case_id": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})
dt_preds_valid.to_pickle(os.path.join(results_dir, "preds_valid_%s.pkl" % dataset_name))


print("\n=============================Start ORF======================================\n")

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "org:resource"
timestamp_col = "time:timestamp"

treatment = 'Treatment'
outcome = 'Outcome' # outcome: 1 or zero

# Prepare data for time of activity treatment
#train, test, valid = train_data, test_data, valid_data
train = train_encoded
test = test_encoded
valid = val_encoded
print("\n train columns")
print(train.columns)
print(test.columns)
print(valid.columns)
#train, valid = train_test_split(train, test_size=0.2, shuffle=False)
features = train.drop([outcome, treatment], axis=1)
features_test = valid.drop([outcome, treatment], axis=1)


Y = train[outcome].to_numpy()
T = train[treatment].to_numpy()
scaler = StandardScaler()
W1 = scaler.fit_transform(features.to_numpy())
#W2 = pd.get_dummies(features[cat_confound_cols]).to_numpy()
W = W1#np.concatenate([W1, W2], axis=1)
X1 = scaler.fit_transform(features.to_numpy())
#X2 = pd.get_dummies(features[cat_hetero_cols]).to_numpy()
X = X1#np.concatenate([X1, X2], axis=1)

X1_test = scaler.fit_transform(features_test.to_numpy())
#X2_test = pd.get_dummies(features_test[cat_hetero_cols]).to_numpy()
X_test = X1_test#np.concatenate([X1_test, X2_test], axis=1)
N_trees = [100]
Min_leaf_size = [50]
Max_depth = [20]
Subsample_ratio = [0.04]
Lambda_reg = [0.01]

for i in itertools.product(N_trees, Min_leaf_size, Max_depth, Subsample_ratio, Lambda_reg):
    print(i)
    n_trees = i[0]
    min_leaf_size = i[1]
    max_depth = i[2]
    subsample_ratio = i[3]
    lambda_reg = i[4]
    est = DMLOrthoForest(n_jobs=-1, backend='threading',
                         n_trees=n_trees, min_leaf_size=min_leaf_size, max_depth=max_depth,
                         subsample_ratio=subsample_ratio, discrete_treatment=True,
                         model_T=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
                         model_Y=Lasso(alpha=lambda_reg),
                         model_T_final=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
                         model_Y_final=WeightedLasso(alpha=lambda_reg),
                         random_state=106
                         )

    ortho_model = est.fit(Y, T, X, W)
    batches = np.array_split(X_test, X_test.shape[0] / 100)
    treatment_effects = est.const_marginal_effect(batches[0])
    ii = 0
    for batch in batches[1:]:
        estimates = est.const_marginal_effect(batch)
        treatment_effects = np.append(treatment_effects, estimates)
        ii += 1
    df_results = valid
    df_results['Treatment Effects'] = treatment_effects
    df_results.to_pickle(os.path.join(results_dir, "df_results_orf_train_%s.pkl" % dataset_name))




print("\n=========================Start ORF Test==========================\n")
# preparing the test set
f_test = test.drop([outcome, treatment], axis=1)
X1_test = scaler.fit_transform(f_test.to_numpy())
#X2_test = pd.get_dummies(f_test[cat_hetero_cols]).to_numpy()
X_test = X1_test

for i in itertools.product(N_trees, Min_leaf_size, Max_depth, Subsample_ratio, Lambda_reg):
    print(i)
    n_trees = i[0]
    min_leaf_size = i[1]
    max_depth = i[2]
    subsample_ratio = i[3]
    lambda_reg = i[4]
    est = DMLOrthoForest(n_jobs=-1, backend='threading',
        n_trees=n_trees, min_leaf_size=min_leaf_size, max_depth=max_depth,
        subsample_ratio=subsample_ratio, discrete_treatment=True,
        model_T=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
        model_Y=Lasso(alpha=lambda_reg),
        model_T_final=LogisticRegression(C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga'),
        model_Y_final=WeightedLasso(alpha=lambda_reg),
        random_state=106
    )

    ortho_model = est.fit(Y, T, X, W)
    batches = np.array_split(X_test, X_test.shape[0] / 100)
    treatment_effects = est.const_marginal_effect(batches[0])
    ii = 0
    for batch in batches[1:]:
        #         print(ii)
        estimates = est.const_marginal_effect(batch)
        treatment_effects = np.append(treatment_effects, estimates)
        ii += 1
    df_results = test
    df_results['Treatment Effects'] = treatment_effects

    # Calculate default (90%) confidence intervals for the default treatment points T0=0 and T1=1
    te_lower, te_upper = est.effect_interval(batches[0])
    ii = 0
    for batch in batches[1:]:
        print(ii)
        lower, upper = est.effect_interval(batch)
        te_lower = np.append(te_lower, lower)
        te_upper = np.append(te_upper, upper)
        ii += 1

    df_results['te_lower'] = te_lower
    df_results['te_upper'] = te_upper
    df_results['Interval Length'] = df_results['te_upper'] - df_results['te_lower']
    df_results.to_pickle(os.path.join(results_dir, "df_results_orf_test_%s.csv" % dataset_name))

    #df_results.to_csv('df_results_orf_test.csv', index=False, sep=';')

#
# #TODO: Start CausalLift
# print("\n======================Start ORF===============================\n")
# train_df = pd.concat([train_encoded, val_encoded],)
# test_df = test_encoded
# del test_encoded
# del train_encoded
# del val_encoded
# gc.collect()
#
#
# train_df.to_csv(os.path.join(results_dir, "train_df_causal.csv"), sep=";", index=False)
# test_df.to_csv(os.path.join(results_dir, "test_df_causal.csv"), sep=";", index=False)
#
# print("\n============Start predictions=================\n")
# train_data = train_df
# test_data = test_df
#
#
# # train the model with pre-tuned parameters
# with open(optimal_params_filename, "rb") as fin:
#     best_params = pickle.load(fin)
#
# y_train = train_data['Outcome']
# del train_data['Outcome']
# del train_data['Treatment']
# X_train = train_data
#
# y_test = test_data['Outcome']
# del test_data['Outcome']
# del test_data['Treatment']
# X_test = test_data
#
# print("Create modle...")
# xgbm = create_model(best_params, X_train, y_train)
#
#
# X_train = xgb.DMatrix(X_train)
# X_test = xgb.DMatrix(X_test)
# print("Predict train...")
# preds_train = xgbm.predict(X_train)  # predictions for train data
# print("Predict test...")
# preds_test = xgbm.predict(X_test)  # predictions for test data
# #import pandas as pd
#
# print("reading prefixes")
# dt_preds_train = pd.DataFrame({"predicted_proba": preds_train, "actual": y_train,})
# dt_preds_train.to_csv(os.path.join(results_dir, "preds_train_%s.csv" % dataset_name), sep=";", index=False)
#
# # write test set predictions
# dt_preds_test = pd.DataFrame({"predicted_proba": preds_test, "actual": y_test})
# dt_preds_test.to_csv(os.path.join(results_dir, "preds_test_%s.csv" % dataset_name), sep=";", index=False)

# print("\n=======================Start Causal Lift=================\n")
#
# print('\n[Estimate propensity scores for Inverse Probability Weighting.]\n')
# cl = CausalLift(train_df, test_df, enable_ipw=True, verbose=3, results_dir=results_dir)  # ,uplift_model_params=uplift_model_params )
#
# print('\n[Create 2 models for treatment and untreatment and estimate CATE (Conditional Average Treatment Effects)]\n')
# train_df, test_df = cl.estimate_cate_by_2_models()
#
# print('\n[Show CATE for train dataset]\n')
# #train_df.to_csv('CATE_for_Train.csv')
# train_df.to_csv(os.path.join(results_dir, "CATE_for_Train.csv"), sep=";", index=False)
#
#
# print('\n[Show CATE for test dataset]\n')
# #test_df.to_csv('CATE_for_Test.csv')
# test_df.to_csv(os.path.join(results_dir, "CATE_for_Test.csv"), sep=";", index=False)
#
#
# print('\n[Estimate the effect of recommendation based on the uplift model]\n')
# estimated_effect_df = cl.estimate_recommendation_impact()

