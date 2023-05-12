import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'edited_biogeme/')
import database as db
import biogeme as bio
from expressions import *
import models
from sklearn.model_selection import KFold
import time
import os


def model_run(raw_data, output_file_path, cv, Re_run, computer_info, model_name, n_jobs, Dependent_var):
    mode_list = list(pd.unique(raw_data[Dependent_var]))
    mode_list = [Dependent_var + '_' + str(i) for i in mode_list]
    base_mode = mode_list[-1]
    if Re_run:
        Results = pd.DataFrame(
            {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
             'n_jobs': [], 'Run_time_5CV_second': [], })
        Results.to_csv(output_file_path, index=False)
    else:
        if not os.path.exists(output_file_path):
            Results = pd.DataFrame(
                {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
                 'n_jobs': [], 'Run_time_5CV_second': [], })
            Results.to_csv(output_file_path, index=False)
        else:
            Results = pd.read_csv(output_file_path)
            flag = 0
            for fold in range(cv):
                if 'Fold' + str(fold+1) not in list(Results['Fold']):
                    flag = 1
                    break
            if flag == 0:
                print('Current model', output_file_path, 'complete, skip it...')
                return
            else:
                Results = pd.DataFrame(
                    {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
                     'n_jobs': [], 'Run_time_5CV_second': [], })
                Results.to_csv(output_file_path, index=False)


    x_columns = list(raw_data.columns)
    x_columns.remove(Dependent_var)
    y_columns = [Dependent_var]
    value = raw_data[Dependent_var].value_counts()
    base = value.max() / len(raw_data)
    X = np.array(raw_data.loc[:, x_columns])
    Y = np.array(raw_data.loc[:, y_columns]).reshape(-1, )
    print('X shape', X.shape)
    print('Y shape', Y.shape)



    x_train_nn_all = X
    #choice_train_nn_all = Y
    tic = time.time()
    kf = KFold(n_splits=cv, random_state=1)

    accuracy = []
    for train_index, test_index in kf.split(x_train_nn_all):
        # if count > 1:
        #     continue
        data = raw_data.loc[train_index,:]

        for ele in mode_list:
            av_name = ele + '_' + 'AV'
            data[av_name] = 1

        database = db.Database("NHTS",data)

        beta_dic = {}
        for mode in mode_list:
            beta_dic[mode] = {}
            asc_name = 'B___' + 'ASC' + '___' + mode
            if mode == base_mode:
                beta_dic[asc_name] = Beta(asc_name + '__mean', 0, None, None, 1)
                # no random for OT
            else:

                ASC_mean = Beta(asc_name + '__mean', 0, None, None, 0)
                ASC_std = Beta(asc_name + '__std', 0, None, None, 0)
                beta_dic[asc_name] = ASC_mean + ASC_std * bioDraws(asc_name, 'NORMAL')

            for name in x_columns:
                if mode == base_mode:
                    beta_name = 'B___' + name + '___' + mode
                    beta_dic[beta_name] = Beta(beta_name, 0, None, None, 1)
                else:
                    beta_name = 'B___' + name + '___' + mode
                    beta_dic[beta_name] = Beta(beta_name, 0, None, None, 0)





        av_variable_dic = {}
        for ele in mode_list:
            av_name = ele + '_' + 'AV'
            av_variable_dic[av_name] = Variable(av_name)
        MODE = Variable(Dependent_var)

        U = {}
        for mode in mode_list:
            asc_name = 'B___' + 'ASC' +'___' +mode
            U[mode] = beta_dic[asc_name]
            for name in x_columns:
                beta_name = 'B___' + name +'___' +mode
                Var = Variable(name)
                U[mode] += beta_dic[beta_name] * Var

        V = {}
        for i in range(len(mode_list)):
            V[i+1] = U[mode_list[i]]

        av = {}
        for i in range(len(mode_list)):
            av_name = mode_list[i] + '_' + 'AV'
            av[i+1] = av_variable_dic[av_name]

        prob = models.logit(V, av, MODE)
        logprob = log(MonteCarlo(prob))

        biogeme = bio.BIOGEME(database, logprob, numberOfThreads=n_jobs, numberOfDraws=200)
        biogeme.modelName = "MixL_classifier"


        ###################
        # if '100k' in model_name:
        #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter = 100, method = 'BFGS')
        # else:
        #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter=100, method='BFGS')
        ############get html for stephane
        biogeme.estimate()


        data_test = raw_data.loc[test_index, :]
        for mode in mode_list:
            col_name = 'exp_U_' + mode
            data_test[col_name] = 0
        for i in range(len(beta_name_list)):
            k = beta_name_list[i]
            v = betas[i]
            mode = k.split('___')[2]
            if '__mean' in mode or '__std' in mode:  # CAR_mean, CAR_std
                mode = mode.split('__')[0]
            col_name = 'exp_U_' + mode
            if 'ASC' in k:
                if 'mean' in k:
                    data_test[col_name] += 1 * v
            else:
                var_name = k.split('___')[1]
                data_test[col_name] += data_test[var_name] * v
        data_test['exp_U'] = 0
        for mode in mode_list:
            col_name = 'exp_U_' + mode
            data_test[col_name] = np.exp(data_test[col_name])
            data_test['exp_U'] += data_test[col_name]

        prob_list = []
        for mode in mode_list:
            col_nameprob = 'prob_' + mode
            prob_list.append(col_nameprob)
            col_name = 'exp_U_' + mode
            data_test[col_nameprob] = data_test[col_name] / data_test['exp_U']

        data_test['max_prob'] = data_test[prob_list].max(axis=1)
        data_test['CHOOSE'] = 0
        choose_list = ['']
        for i in range(len(mode_list)):
            col_nameprob = 'prob_' + mode_list[i]
            data_test.loc[data_test[col_nameprob] == data_test['max_prob'], 'CHOOSE'] = i + 1

        acc = len(data_test.loc[data_test['CHOOSE'] == data_test[Dependent_var]]) / len(data_test)
        accuracy.append(acc)

        print('accuracy:', acc)

    Training_time = round((time.time() - tic), 2)

    for cv_num in range(len(accuracy)):
        Results = pd.concat([Results, pd.DataFrame(
            {'Model': [model_name], 'Fold': ['Fold' + str(cv_num + 1)], 'Accuracy': [accuracy[cv_num]],
             'Computer_info': [computer_info], 'n_jobs': [n_jobs],
             'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)
    # save in every iteration
    avg_acc = np.mean(accuracy)
    Results = pd.concat([Results, pd.DataFrame(
        {'Model': [model_name], 'Fold': ['Average'], 'Accuracy': [avg_acc], 'Computer_info': [computer_info],
         'n_jobs': [n_jobs],
         'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)


    Results.to_csv(output_file_path, index=False)

if __name__ == '__main__':

    tic = time.time()
    # Parameters:

    sample_size_list = ['1k'] #Mixed logit, only for 1k samples
    # sample_size_list = ['10k']
    # sample_size_list = ['1k','10k','100k']
    computer_info = 'I9-9900K'

    Re_run = True # True: rerun all models, False: if there are results existed, jump it
    n_jobs = 1
    # Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']
    Dependent_var_list = ['MODE']

    Estimator_name = 'MixL_biogeme'
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
                if sample_size != '1k':
                    print('Mix logit not work for', sample_size)
                    continue
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
                print('Total running time:', round(time.time() - tic, 1),'s')
                print('Current model', output_file_path)


                model_run(data, output_file_path, 5, Re_run, computer_info, Estimator_name, n_jobs, Dependent_var)







