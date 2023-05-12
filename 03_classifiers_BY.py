'''
Main sample code:
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format
https://sebastianraschka.com/faq/docs/tensorflow-vs-scikitlearn.html
'''
import numpy as np
import pandas as pd
import time
import os
import copy
import matplotlib.pyplot as plt
# process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# DA models

# Bayes Models
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
#
from sklearn.model_selection import cross_val_score


from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


class Classifier:
    def __init__(self, raw_data, Output_name, Dependent_var,Re_run, computer_info, n_jobs, models):
        self.raw_data = raw_data
        self.Output_name = Output_name
        self.Dependent_var = Dependent_var
        self.computer_info = computer_info
        self.n_jobs = n_jobs
        self.Re_run = Re_run
        self.model = models


    def data_precocess(self):
        y_columns = [self.Dependent_var]
        x_columns = list(self.raw_data.columns)
        x_columns.remove(self.Dependent_var)
        value = self.raw_data[self.Dependent_var].value_counts()
        self.base = value.max()/len(self.raw_data)
        self.X = np.array(self.raw_data.loc[:,x_columns])
        self.Y = np.array(self.raw_data.loc[:,y_columns]).reshape(-1,)

        print('X shape', self.X.shape)
        print('Y shape',self.Y.shape)


    def excute(self):
        # process data
        tic = time.time()
        self.data_precocess()
        print ('process data time:', round(time.time() - tic,1))
        # define model


        # intial output table

        output_file_path = self.Output_name
        if self.Re_run:
            Results = pd.DataFrame(
                {'Model': [], 'Fold': [], 'Accuracy': [],  'base': [], 'Computer_info': [],
                 'n_jobs': [], 'Run_time_5CV_second': [],})
            Results.to_csv(output_file_path, index=False)
        else:
            if not os.path.exists(output_file_path):
                Results = pd.DataFrame(
                    {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
                     'n_jobs': [], 'Run_time_5CV_second': [], })
                Results.to_csv(output_file_path, index=False)
            else:
                Results = pd.read_csv(output_file_path)
        # train models
        Existing_models = list(Results['Model'])
        Model_dict = {**self.model}
        print (Model_dict)
        for name in Model_dict:
            try:
                if not self.Re_run:
                    if name in Existing_models:
                        print(name, 'exist, skip...')
                        continue
                model = Model_dict[name]
                if name == 'NB_MultiModal':
                    if len(self.X[self.X<0]) > 0:
                        X_input = copy.deepcopy(self.X)
                        Y_input = copy.deepcopy(self.Y)
                        X_input = MinMaxScaler().fit_transform(X_input)
                    else:
                        X_input = copy.deepcopy(self.X)
                        Y_input = copy.deepcopy(self.Y)
                else:
                    X_input = copy.deepcopy(self.X)
                    Y_input = copy.deepcopy(self.Y)
                tic = time.time()
                print("Training model ", name, " ...")
                accuracy = cross_val_score(model, X=X_input, y=Y_input, cv=5, n_jobs=self.n_jobs, error_score='raise')
                Training_time = round((time.time() - tic), 2)
                for cv_num in range(len(accuracy)):
                    Results = pd.concat([Results, pd.DataFrame({'Model':[name],'Fold':['Fold'+str(cv_num+1)],'Accuracy': [accuracy[cv_num]],'Computer_info':[self.computer_info],'n_jobs':[self.n_jobs],
                                             'base':[self.base],'Run_time_5CV_second':[Training_time]})],sort=False)
                # save in every iteration
                avg_acc = np.mean(accuracy)
                Results = pd.concat([Results, pd.DataFrame(
                    {'Model': [name],'Fold': ['Average'],'Accuracy': [avg_acc], 'Computer_info': [self.computer_info], 'n_jobs': [self.n_jobs],
                       'base': [self.base], 'Run_time_5CV_second': [Training_time]})], sort=False)
                Results.to_csv(output_file_path,index=False)
            except:
                print('Model', name, 'is not applicable')
                raise


if __name__ == '__main__':
    tic = time.time()
    # Parameters:


    sample_size_list = ['1k','10k','100k']
    # sample_size_list = ['10k']

    computer_info = 'I9-9900K'

    Re_run = True # True: rerun all models, False: if there are results existed, jump it
    n_jobs = 1
    # Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']
    Dependent_var_list = ['MODE']

    models = {'NB_Ber': BernoulliNB(), 'NB_Gau': GaussianNB(), 'NB_MultiModal': MultinomialNB()}
    Estimator_name = 'BY_python'
    # DATASET_list = ['NHTS', 'London', 'SG']
    DATASET_list = ['NHTS']
    for DATASET in DATASET_list:
        for Dependent_var in Dependent_var_list:
            if DATASET == 'London' or DATASET == 'SG':
                if Dependent_var != 'MODE':
                    continue
            if Dependent_var=='MODE':
                output_name = 'MC'
                data_name = 'mode_choice'
            elif Dependent_var=='CAR_OWN':
                output_name = 'CO'
                data_name = 'car_ownership'
            else:
                output_name= 'TP'
                data_name = 'trip_purpose'

            for sample_size in sample_size_list:
                if DATASET == 'SG':
                    if sample_size == '100k':
                        continue

                if DATASET == 'London':
                    output_file_path = 'Results/Results_London_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    data_name_read = 'data_London_' + data_name + '_' + sample_size + '.csv'
                    data = pd.read_csv('London_dataset/' + data_name_read)
                elif DATASET == 'SG':
                    output_file_path = 'Results/Results_SG_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    data_name_read = 'data_SG_' + data_name + '_' + sample_size + '.csv'
                    data = pd.read_csv('SG_dataset/' + data_name_read)
                else:
                    output_file_path = 'Results/Results_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    data_name_read = 'data_' + data_name + '_' + sample_size + '.csv'
                    data = pd.read_csv('data/' + data_name_read)
                print('Read raw data time:', round(time.time() - tic, 1),'s')

                m = Classifier(data, output_file_path, Dependent_var, Re_run, computer_info,n_jobs, models)
                m.excute()