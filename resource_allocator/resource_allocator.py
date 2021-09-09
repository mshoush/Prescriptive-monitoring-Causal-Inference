#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
import numpy as np
import random
import pickle5 as pickle
import os


# In[ ]:


def read_pickle_data(file):
    with open(file, "rb") as fh:
        data = pickle.load(fh)
    
    return data
    


# In[ ]:


orf_test = read_pickle_data("./predictive_and_prescriptive/orf_test.pkl")
orf_test


# In[ ]:


# Resource allocator
import time
import threading
import random


def allocateRes(res, dist): # Allocate
    t = threading.Thread(target=block_and_release_res, args=(res,dist))
    t.start()    
    print("Apply treatment")
    print(f"resource number: {res} blocked")
    print(f"strat timer for resource number: {res}")
        
    
def block_and_release_res(res,dist): # timer
#     t_dists = ["normal", "fixed", "exp"]
    t_dist=dist
    if t_dist =="normal":
        treatment_duration = int(random.uniform(1, 60))
    elif t_dist=="fixed":
        treatment_duration = 30#int(np.random.exponential(60, size=1))
    else:
        treatment_duration = int(np.random.exponential(60, size=1))    
    
    
    #treatment_duration = int(random.uniform(1, 60))
    time.sleep(treatment_duration)
    print(f"Treatment duration is: {treatment_duration}, for resource number: {res}")
    print(f"Release res: {res}")
    nr_res.append(res)
    print("Do more stuff here")
    print("")



# In[ ]:


cu = 20 # argv[1]
ct = 1  # argv[2]

resources = 10 # argv[3]
res_list = list(range(1, resources+1, 1))

list_gains = []

orf_test['cost_un'] = 0
orf_test['cost_treat'] = 0
orf_test['gain'] = 0

t_dist="normal" # argv[5]
prob_threshold = 0.5 #argv[6]


#t_dist = dist
if t_dist =="normal":
    folder = "results_normal_dur/"
elif t_dist=="fixed":
    folder = "results_fixed_dur/"
else:
    folder = "results_exp_pos_dur/"

results_dir = "./../results/evaluation/"+folder
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

for i in range(1, len(res_list)+1):
    nr_res = res_list[0:i]
    
    df_gains_without_res_2_1 = pd.DataFrame()
    df_gains_with_res_2_1 = pd.DataFrame()

    for row in orf_test.iterrows():
        pred_proba = row[-1][-8]

        if pred_proba>prob_threshold:  
            # cost un
            row[-1][-3] = pred_proba * cu                        
            # CATE
            cate = row[-1][-12]
            # cost treat
            row[-1][-2] = (np.subtract(pred_proba, cate )) * cu + ct
            # gain
            row[-1][-1] = row[-1][-3] - row[-1][-2]

            if row[-1][-1]>0:
                df_gains_without_res_2_1 = df_gains_without_res_2_1.append(list(row)[1],)
                list_gains.append([row[-1][-1]])
                sorted_list = sorted(list_gains, key=lambda x: x[0],reverse=True)
                if sorted_list:                
                    max_gain = sorted_list[0][0]
                    if max_gain > 0 and nr_res:
                        df_gains_with_res_2_1 = df_gains_with_res_2_1.append(list(row)[1],)
                        res = nr_res[0]
                        nr_res.remove(nr_res[0])
                        print(f"allocate res: {res}")
                        allocateRes(res, t_dist)
                        print("allocateRes func has returned")
                        print("") 

                else:
                    pass
            else:
                pass
    
    df_gains_without_res_2_1.to_csv(results_dir+'df_gains_without_res_2_'+str(nr_res[-1])+'.csv', index=False, sep=';')
    df_gains_with_res_2_1.to_csv(results_dir+'df_gains_with_res_2_'+str(nr_res[-1])+'.csv', index=False, sep=';')

