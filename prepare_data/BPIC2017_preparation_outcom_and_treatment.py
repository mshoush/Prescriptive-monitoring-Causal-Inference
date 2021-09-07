#!/usr/bin/env python
# coding: utf-8

# # BPIC2017 (Application to Approval business proces)
# * As part of the BPIC 2017 [1] researches were offered to conduct an analysis of the process of issuing credit offers in a financial institute. The data were two sets for the period from the beginning of 2016 to February 2, 2017.
# 
# * Application event log (AL) contains data describing the whole process under consideration from filling out a `loan application to decision-making` (approving or declining). It contains 1 202 267 unique event records and 31 509 unique cases.

# ## General description of process scenarios:
# 
# * All cases of the process under consideration can be divided into three categories:
# 1. `Successful completion:` process case includes “A Pending” event;
# 2. `Denied by bank:` process case includes “A Denied” event;
# 3. `Cancelled by client:` process case includes “A Cancelled” event.
# 
# ## Outcome definition: 
# * we define cases that have `A Pending` event as a positive outcome (or regular case) with encoded value equal to $0$ in the log. 
# * we define cases that have `A Denied` or `A Cancelled` events as a negative outcome (or deviant case) with encoded value equal to $1$ in the log.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
import pickle5 as pickle

# In[2]:


# read data and add label
file = argv[1]

def read_data(file):
    # read original data
    #df = pd.read_csv(file, sep =';')
    with open(file, "rb") as fh:
        df = pickle.load(fh)

    #df = pd.read_pickle(file)
    # delte old label
    del df['label']    
    return df



def add_label(file):    
    # function to cheack if a case contains A_Pending activity or not
    def check_A_Pending(gr):
        df = pd.DataFrame(gr)
        if  "A_Pending" in list(df['Activity']):
            df['label'] = "regular" 
        else:
            df['label'] = "deviant" 
        return df
    
    # read data and del old label
    df = read_data(file)
    
    # add new label for each case based on A_Pending
    df = df.groupby('Case ID').apply(check_A_Pending)
    df = df.reset_index(drop=True)
    return df

# fn to add #offers column to the original data based on "O_Created" Activity
def add_No_Of_Offers(df):
    # get all observations with O_Created activity
    tmp_df = df[df['Activity'] == "O_Created"]  # to count offers
    
    # count numbe rof offers for each case
    tmp_df2 = pd.DataFrame(tmp_df.groupby(['Case ID'])['Activity'].count()).reset_index()
    tmp_df2.columns = ['Case ID', 'NumberOfOffers']
    df = pd.merge(tmp_df2, df, on='Case ID')
    return df 
    
# function to add treatment based on the number of offers
def add_treatment(df):
    
    # function to cheack if a case contains A_Pending activity or not
    def check_NoOfOffers(gr):
        df = pd.DataFrame(gr)
        
        # case should be not treated if it receives more than one offer
        if  list(df['NumberOfOffers'])[0] <= 1:
            df['treatment'] = "treat"  # T=1
        else:
            df['treatment'] = "noTreat" # T=0
        return df
    

    # add new treatment for each case based on number of offers
    # cases with only one offer should be treated
    df = df.groupby('Case ID').apply(check_NoOfOffers)
    df = df.reset_index(drop=True)
    return df

df = add_label(file)
df = add_No_Of_Offers(df)
df = add_treatment(df)



# In[6]:


# Save the prepared log for predicitve and prescriptive models
#df.to_csv('bpic2017_A_pending_treat.csv', index=False, sep=';')
#import pickle5 as pickle
#with open("prepared_bpic2017.pkl", "rb") as fh:
df.to_pickle("./predictive_and_prescriptive/prepared_bpic2017.pkl")
#
# data = pickle.load(fh)
# df.to_pickle("./predictive_and_prescriptive/prepared_bpic2017.pkl")


# In[ ]:




