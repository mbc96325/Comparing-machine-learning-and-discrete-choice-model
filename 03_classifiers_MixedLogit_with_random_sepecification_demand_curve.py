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


def model_run(raw_data, data_test_raw, col_cost, num_points, MODE_WANT, output_file_path, cv, Re_run, computer_info, model_name, n_jobs, Dependent_var):
    mode_list = list(pd.unique(raw_data[Dependent_var]))
    mode_list = [Dependent_var + '_' + str(i) for i in mode_list]
    mode_list = sorted(mode_list)

    # if Re_run:
    #     Results = pd.DataFrame(
    #         {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
    #          'n_jobs': [], 'Run_time_5CV_second': [], })
    #     Results.to_csv(output_file_path, index=False)
    # else:
    #     if not os.path.exists(output_file_path):
    #         Results = pd.DataFrame(
    #             {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
    #              'n_jobs': [], 'Run_time_5CV_second': [], })
    #         Results.to_csv(output_file_path, index=False)
    #     else:
    #         Results = pd.read_csv(output_file_path)
    #         flag = 0
    #         for fold in range(cv):
    #             if 'Fold' + str(fold+1) not in list(Results['Fold']):
    #                 flag = 1
    #                 break
    #         if flag == 0:
    #             print('Current model', output_file_path, 'complete, skip it...')
    #             return
    #         else:
    #             Results = pd.DataFrame(
    #                 {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
    #                  'n_jobs': [], 'Run_time_5CV_second': [], })
    #             Results.to_csv(output_file_path, index=False)


    x_columns = list(raw_data.columns)
    x_columns.remove(Dependent_var)
    y_columns = [Dependent_var]
    value = raw_data[Dependent_var].value_counts()
    base = value.max() / len(raw_data)
    X = np.array(raw_data.loc[:, x_columns])
    Y = np.array(raw_data.loc[:, y_columns]).reshape(-1, )
    print('X shape', X.shape)
    print('Y shape', Y.shape)

    X_test = np.array(data_test_raw.loc[:, x_columns])
    Y_test = np.array(data_test_raw.loc[:, y_columns]).reshape(-1, )
    col_cost_id = x_columns.index(col_cost)

    x_train_nn_all = X
    #choice_train_nn_all = Y

    #kf = KFold(n_splits=cv, random_state=1)

    accuracy = []



    # specification
    num_mode_put = len(mode_list) - 1 #
    assert num_mode_put  <= len(mode_list) - 1
    initial_seed = 100
    specification = {}
    for mode in mode_list:
        specification[mode] = []

    for name in x_columns:
        initial_seed += 1
        # every x can only be feed into num_mode_put modes
        np.random.seed(initial_seed)
        modes_used = np.random.choice(mode_list,size =  num_mode_put,replace=False)
        for mode in modes_used:
            beta_name = 'B___' + name + '___' + mode
            specification[mode].append(beta_name)
        a=1


    html_name = output_file_path.replace('Demand_curve_results/','Biogeme_html_file_demand_curve/')
    html_name = html_name.replace('.csv', '')

    count = 0

    write_html_time_total = 0
    tic = time.time()

        # count += 1
        # if count > 1:
        #     continue
    data = raw_data.copy()

    for ele in mode_list:
        av_name = ele + '_' + 'AV'
        data[av_name] = 1

    database = db.Database("NHTS",data)
    html_name_cv = html_name + '_CV_' + str(int(count))



    beta_dic = {}
    initial_seed += 1
    np.random.seed(initial_seed)
    modes_used = np.random.choice(mode_list, size=num_mode_put, replace=False)
    for mode in modes_used:
        asc_name = 'B___' + 'ASC' + '___' + mode
        ASC_mean = Beta(asc_name + '__mean', 0, None, None, 0)
        ASC_std = Beta(asc_name + '__std', 0, None, None, 0)
        beta_dic[asc_name] = ASC_mean + ASC_std * bioDraws(asc_name, 'NORMAL')





    av_variable_dic = {}
    for ele in mode_list:
        av_name = ele + '_' + 'AV'
        av_variable_dic[av_name] = Variable(av_name)
    MODE = Variable(Dependent_var)

    U = {}
    for mode in mode_list:
        U[mode] = 0
        asc_name = 'B___' + 'ASC' + '___' + mode
        if asc_name in beta_dic:
            U[mode] += beta_dic[asc_name]

        for name in x_columns:
            beta_name = 'B___' + name + '___' + mode
            if beta_name in specification[mode]:  # only specification
                Var = Variable(name)
                beta_dic[beta_name] = Beta(beta_name, 0, None, None, 0)
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

    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=n_jobs, numberOfDraws=300)
    biogeme.modelName = html_name_cv


    ###################
    # if '100k' in model_name:
    #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter = 100, method = 'BFGS')
    # else:
    #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter=100, method='BFGS')
    ############get html for stephane
    # biogeme.estimate()


    betas, beta_name_list, write_html_time = biogeme.estimate_baichuan(max_iter=None, method='BFGS',Write_html=True) # max_iter = None means no max_iter used
    write_html_time_total += write_html_time



    #############PREDICTION

    Results = pd.DataFrame()


    min_X = np.min(X_test[:, col_cost_id])
    max_X = np.max(X_test[:, col_cost_id])

    all_points = np.linspace(min_X, max_X, num=num_points)

    prop_list = []

    for point_value in all_points:
        data_test = data_test_raw.copy()
        data_test.loc[:,col_cost] = point_value

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


        y_prob = data_test.loc[:,'prob_MODE_' + str(MODE_WANT)].values
        prop = np.mean(y_prob)

        # y_pred = data_test['CHOOSE'].values
        # prop = sum(y_pred == MODE_WANT) / len(y_pred)


        prop_list.append(prop)

        # acc = len(data_test.loc[data_test['CHOOSE'] == data_test[Dependent_var]]) / len(data_test)
        # accuracy.append(acc)

        # print('accuracy:', acc)

    Results = Results.append(
        pd.DataFrame({'Model': [model_name] * num_points, 'x_values': all_points, 'pred_prop': prop_list}))


    # Training_time = round((time.time() - tic - write_html_time_total), 2)
    #
    # for cv_num in range(len(accuracy)):
    #     Results = pd.concat([Results, pd.DataFrame(
    #         {'Model': [model_name], 'Fold': ['Fold' + str(cv_num + 1)], 'Accuracy': [accuracy[cv_num]],
    #          'Computer_info': [computer_info], 'n_jobs': [n_jobs],
    #          'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)
    # # save in every iteration
    # avg_acc = np.mean(accuracy)
    # Results = pd.concat([Results, pd.DataFrame(
    #     {'Model': [model_name], 'Fold': ['Average'], 'Accuracy': [avg_acc], 'Computer_info': [computer_info],
    #      'n_jobs': [n_jobs],
    #      'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)


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

    Estimator_name = 'MixL_biogeme_RandSpec'
    # DATASET_list = ['NHTS', 'London', 'SG']
    DATASET_list = ['London']


    col_cost = 'dur_pt_bus' # 'cost_driving_fuel' #
    num_points = 100

    MODE_WANT = 3 #4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive
    # (1: walk, 2: cycle, 3: public transport, 4:drive)

    ###############
    # TO BE REVISED
    ###############


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
                # if sample_size != '1k':
                #     print('Mix logit not work for', sample_size)
                #     continue
                if DATASET == 'SG':
                    if sample_size == '100k':
                        continue

                if DATASET == 'London':

                    tail_name = ''
                    if sample_size == '1k':
                        tail_name = '_1k'

                    output_file_path = 'Demand_curve_results/Results_London_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    data_name_read = 'train_data_London' + tail_name + '.csv'
                    data_test_read = 'test_data_London' + tail_name + '.csv'
                    data = pd.read_csv('London_dataset/' + data_name_read)
                    data_test = pd.read_csv('London_dataset/' + data_test_read)


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

                output_file_path = output_file_path.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
                    int(MODE_WANT)) + '.csv')

                model_run(data, data_test, col_cost, num_points, MODE_WANT, output_file_path, 5, Re_run, computer_info, Estimator_name, n_jobs, Dependent_var)







