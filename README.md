# Prescriptive Process Monitoring Under Resource Constraints: A Causal Inference Approach

This project contains supplementary material for the article "[Prescriptive Process Monitoring Under Resource Constraints: A Causal Inference Approach](https://arxiv.org/abs/2109.02894)" by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en), and [Marlon Dumas](https://kodu.ut.ee/~dumas/).


The approach combines a predictive model to identify cases that are likely to end in a negative outcome (and hence create a cost) with a causal
model to determine which cases would most benefit from an intervention in their current state. These two models are embedded into an allocation procedure that
allocates resources to case interventions based on their estimated net gain.

# Dataset: 
Dataset can be found in the "prepare_data" folder or on the following link.
* [BPIC2017, ie., a loan application process.](https://drive.google.com/file/d/1w1MPzU7Rz-wTYcSkLqWyGZJOI_RlYGzS/view?usp=sharing)



# Reproduce results:
To reproduce esults, please run the following:

* First use the following command to install required packages from a venv. 
                                    
                                     conda env create -f venv.yml

* Next, please execute the following notebook to run the all experiments. 

                                     run_experiments.ipynb
                                     
                                     


                 


