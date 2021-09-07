# Prescriptive Process Monitoring Under Resource Constraints: A Causal Inference Approach

# Improved methods for Predictive Process Monitoring:
This project is an extension to work done by [irhete/predictive-monitoring-thesis ](https://github.com/irhete/predictive-monitoring-thesis) as a master thesis at the University of Tartu, Estonia. In this thesis, we introduced three different methods to improve the currently existing techniques of outcome-oriented predictive process monitoring. 

This repository and our contributions are organized as follows:
* **Contribution 1:** Adding CatBoost method to outcome-oriented PPM techniques. 
* **Contribution 2:** Adding a new complex sequence encoding technique on the basis of discrete wavelet transform and neural networks. 
* **Contribution 3:** Adding Inter-case features to the experiments.  

# Dataset: 
You need to download below datasets and modify the path in `dataset_confs.py` script. 

* [Labelled datasets](https://drive.google.com/drive/folders/1ut9HR5I4Bvo96WcG09Boex_XfC6rJujZ?usp=sharing)
* [Inter-case features datasets](https://drive.google.com/drive/folders/1E26I981qyMNj1laTNKoCCC_PneGzlT5R?usp=sharing)




# Reproduce results:
* To run this project, you need to install packages from `requirements.txt` file, and to do so run the below command:                             

                  conda create --name env_name --file requirements.txt --yes
                  
* Above command will create a virtual environment, so you need to activate it afterwards using below command:

                  conda activate env_name
                  
* To reproduce results for each contribution you need to go inside the corresponding folder, and then run the following commands: 
                
    1. Hyperparameter optimization:
      
                                python experiments_optimize_params.py <data set> <bucketing_encoding> <classifier> <nr_iterations>
                                
    2. Training and evaluating final models: 
      
                                python experiments.py <data set> <bucketing_encoding> <classifier>
                                
    3. Execution times of final models: 
      
                                python experiments_performance.py <data set> <bucketing_encoding> <classifier> <nr_iterations>

* `For Example_CatBoost`: 
                  
                  cd ./CatBoost/
                  python experiments_optimize_params.py production single_laststate catboost 1
                  python experiments.py production single_laststate catboost 
                  python experiments_performance.py production single_laststate catboost 1
                  
                  
 * `For Example_Wavelet`: 
                  
                  cd ./Wavelet/
                  python experiments_optimize_params.py production single_waveletLast catboost 1
                  python experiments.py production single_waveletLast catboost 
                  python experiments_performance.py production single_waveletLast catboost 1
                  

* `For Example_Inter-case features`: 
                  
                  cd ./Inter-case_features/
                  python experiments_optimize_params.py production single_laststate catboost 1
                  python experiments.py production single_laststate catboost 
                  python experiments_performance.py production single_laststate catboost 1


          


                   


