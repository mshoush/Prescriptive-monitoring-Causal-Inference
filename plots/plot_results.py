#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.ticker as mtick


tqdm.pandas()
pd.set_option('display.max_columns', None)


from statsmodels.stats.proportion import proportions_ztest

import sklearn as sk
from sklearn.metrics import auc
import xgboost as xgb

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle


get_ipython().run_line_magic('matplotlib', 'inline')



# In[2]:


def read_pickle_data(file):
    with open(file, "rb") as fh:
        data = pickle.load(fh)
    
    return data
    
orf_test = read_pickle_data("./../predictive_and_prescriptive/orf_test.pkl")
orf_test


# In[3]:


total_nu_cases = len(set(orf_test['case_id']))


# In[4]:


def read_data(results):
    # normal distribution
    df_1 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_1.csv", sep=';')
    df_1.name = "10%"

    df_2 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_2.csv", sep=';')
    df_2.name = "20%"

    df_3 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_3.csv", sep=';')
    df_3.name = "30%"

    df_4 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_4.csv", sep=';')
    df_4.name = "40%"

    df_5 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_5.csv", sep=';')
    df_5.name = "50%"

    df_6 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_6.csv", sep=';')
    df_6.name = "60%"

    df_7 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_7.csv", sep=';')
    df_7.name = "70%"

    df_8 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_8.csv", sep=';')
    df_8.name = "80%"

    df_9 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_9.csv", sep=';')
    df_9.name = "90%"

    df_10 = pd.read_csv("./../results/"+results+"/df_gains_with_res_2_10.csv", sep=';')
    df_10.name = "100%"
    
    dfs = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10]
    return dfs


# In[5]:


gains = []
names = []
number_cases = []
percent_cases = []
#df_res = []


def get_names_gains_percentCases(dfs):
    dfs = dfs
    
    for df in dfs:
        #df_res.append(df)
        gains.append(sum(df['CATE']))
        names.append(df.name)
        number_cases.append(len(set(list(df['case_id']))))
        percent_cases.append((np.round((len(set(list(df['case_id']))) * 100) / total_nu_cases, 2)))
        print(f"data is: {df.name}, and number of treated cases are: {np.round((len(set(list(df['case_id']))) * 100) / total_nu_cases, 2)}        with gain: {sum(df['gain'])}")
    return names, gains, percent_cases,# df_res


# In[6]:


def run_plot(names, gains, percent_cases, typee="None"):
    plt.figure()
    names, gains, percent_cases = names, gains, percent_cases
    
    df = pd.DataFrame({"date":names,
                       "gains": gains, 
                       "treated cases": percent_cases})

    ax = df.plot(x="date", y="gains", legend=False,  marker="o")
    ax.spines['left'].set_color('#105b7e')
    ax.tick_params(axis='y', color='#105b7e', labelcolor='#105b7e')
    ax.set_ylabel('Total Uplift', color="#105b7e")
    ax.set_xlabel('% of availble resources',)
    #plt.yscale("log",  base=2)

    ax2 = ax.twinx()
    df.plot(x="date", y="treated cases", ax=ax2, legend=False, color="#c6511a",  marker="o")
    ax2.set_ylabel('% of treated cases', color="#c6511a")
    ax2.spines['right'].set_color('#c6511a')
    ax2.tick_params(axis='y', color='#c6511a', labelcolor='#c6511a')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.legend(bbox_to_anchor=(0.25, 1))
    ax2.legend(bbox_to_anchor=(0.37, 0.9))
    ax.grid(True)
    plt.tight_layout()
    plt.title(typee+" Distribution")

    #plt.title("Impact of available resources on the gain and treated cases")
    plt.savefig(typee+".png")
    plt.show()
    return df


# In[7]:


import time
#df_res = []
df_res = []
prob_threshold = 0.5

# RQ1 and RQ2
dirr_ev = ["evaluation/results_normal_dur", "evaluation/results_fixed_dur","evaluation/results_exp_pos_dur"]# "results_exp_pos_dur"]

# RQ3
dirr_baseline = ["baseline/"+str(prob_threshold)+"/results_normal_dur", "baseline/"+str(prob_threshold)+"/results_fixed_dur","baseline/"+str(prob_threshold)+"/results_exp_pos_dur"]# "results_exp_pos_dur"]

types = ["Normal", "Fixed", "Exponential"]

i=0

# RQ3, change "dirr_baseline" to "dirr_ev" for RQ1 and RQ2
for file in dirr_baseline:    
    dfs = read_data(file)
    names, gains, percent_cases, = get_names_gains_percentCases(dfs)
    #df_res.append(df)
    df =run_plot(names, gains, percent_cases, typee=types[i])
    df_res.append(df)
    names, gains, percent_cases =[],[],[]
    i+=1


# In[8]:


df_normal = df_res[0]
df_normal = df_normal.rename(columns={"date": "x_no", "gains": "gain_normal", "treated cases": 'normal'})
df_fix = df_res[1]
df_fix = df_fix.rename(columns={"date": "x", "gains": "gain_fixed", "treated cases": 'fixed'})
df_exp = df_res[2]
df_exp = df_exp.rename(columns={"date": "x", "gains": "gain_exponential", "treated cases": 'exponential'})


# In[9]:


df_final = pd.concat([df_normal, df_fix, df_exp], axis=1)
df_final


# In[10]:


plt.figure()
colors = [ '#2874a6','#148f77',   "#b03a2e" ]
colors2 = ["#839192", "#2e4053", "#884ea0"]


df = df_final

ax = df.plot(x="x_no", y=[ "gain_fixed","gain_normal", "gain_exponential"],color=colors,
             legend=False,  marker="o")
ax.tick_params(axis='y', )#color='#105b7e', labelcolor='#105b7e')
ax.set_ylabel('Total gain',)# color="#105b7e")
ax.set_xlabel('% of availble resources',)

ax2 = ax.twinx()
df.plot(x="x_no", y=["fixed","normal",  "exponential"], ax=ax2, legend=False, 
        color=colors2,  marker="o")
ax2.set_ylabel('% of treated cases',)# color="#c6511a")
ax2.tick_params(axis='y',)# color='#c6511a', labelcolor='#c6511a')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="upper left", ncol=len(df.columns),title="Total gian",  prop={'size': 7.8},)
ax2.legend( loc='lower right', title="% Treated cases",prop={'size': 8})
ax.grid(True)
plt.tight_layout()

plt.savefig("RQ306.pdf",  bbox_inches='tight')
plt.show()

