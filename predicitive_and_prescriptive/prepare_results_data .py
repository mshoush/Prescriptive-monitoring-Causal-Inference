#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv

import pickle5 as pickle


# In[2]:


def read_pickle_data(file):
    with open(file, "rb") as fh:
        data = pickle.load(fh)
    
    return data
    


# In[3]:


df_test_prefix = read_pickle_data("/home/mshoush/ut_cs_phd/phd/code/gitHub/predictive_and_prescriptive/results_dir/dt_test_prefixes.pkl")
preds_test = read_pickle_data("/home/mshoush/ut_cs_phd/phd/code/gitHub/predictive_and_prescriptive/results_dir/preds_test_prepared_bpic2017.pkl")
#preds_test


# In[4]:


# Predictive part
preds_test["prefix_nr"]= list(df_test_prefix.groupby("Case ID").first()["prefix_nr"])
preds_test["case_id"]= list(df_test_prefix.groupby("Case ID").first()["orig_case_id"])
preds_test['time:timestamp'] = list(df_test_prefix.groupby("Case ID").last()["time:timestamp"])
preds_test = preds_test.sort_values(by=['time:timestamp']).reset_index(drop=True)
preds_test


# In[5]:


# ORF part
df_results_test_orf = read_pickle_data("/home/mshoush/ut_cs_phd/phd/code/gitHub/predictive_and_prescriptive/results_dir/df_results_orf_test_prepared_bpic2017.pkl")
df_results_test_orf.rename(columns={'Treatment Effects':'CATE'}, inplace=True)

df_results_test_orf


# In[6]:


# Combine predictive and causal models
orf_test = pd.concat([df_results_test_orf, preds_test], axis=1)
orf_test.to_pickle("orf_test.pkl")
orf_test

