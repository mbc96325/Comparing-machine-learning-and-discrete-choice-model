import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import time
import os
import sys
sys.path.insert(0,'edited_biogeme/')
import database as db
import biogeme as bio
import models
from expressions import *



def predict_NL(betas, biogeme_file, data, modes_list, Dependent_var):
    for ele in modes_list:
        av_name = ele + '_' + 'AV'
        data[av_name] = 1
    database = db.Database("NL_test", data)
    # The choice model is a nested logit

    prob = {}
    for idx in range(len(modes_list)):
        prob_name = 'prob_' + modes_list[idx]
        prob[prob_name] = models.nested(biogeme_file['V'], biogeme_file['av'], biogeme_file['nests'], idx+1)

    simulate = {}
    for idx in range(len(modes_list)):
        prob_name = 'prob_' + modes_list[idx]
        simulate[prob_name] = prob[prob_name]


    biogeme = bio.BIOGEME(database, simulate)


    # Extract the values that are necessary
    betaValues = betas
    # simulatedValues is a Panda dataframe with the same number of rows as
    # the database, and as many columns as formulas to simulate.
    simulatedValues = biogeme.simulate(betaValues)

    prob_list = list(simulatedValues.columns)
    data_test = data
    for key in prob_list:
        data_test[key] = 0
    data_test.loc[:,prob_list] = simulatedValues.loc[:, prob_list]
    data_test['max_prob'] = data_test[prob_list].max(axis=1)
    data_test['CHOOSE'] = 0
    for idx in range(len(modes_list)):
        col_nameprob = 'prob_' + modes_list[idx]
        data_test.loc[data_test[col_nameprob]==data_test['max_prob'],'CHOOSE'] = idx+1

    acc = len(data_test.loc[data_test['CHOOSE']==data_test[Dependent_var]])/len(data_test)
    return acc, data_test

def model_run(raw_data, data_test_raw, col_cost, num_points, MODE_WANT, output_file_path, cv, Re_run, computer_info, model_name, n_jobs, Dependent_var, DATASET):
    # 1 Home based other 2 home based shop 3 home based social 4 home based work 5 non home based
    if Dependent_var == 'TRIPPURP' and DATASET == 'NHTS':
        nest_1 = [1,2,3] # need to revise for different depend var
        nest_2 = [4,5]
    elif Dependent_var == 'MODE' and DATASET == 'NHTS':
        nest_1 = [2,3,4] # private car
        nest_2 = [1,5,6]
    elif Dependent_var == 'CAR_OWN' and DATASET == 'NHTS':
        nest_1 = [1,2] # low car ownership
        nest_2 = [3,4,5]
    elif Dependent_var == 'MODE' and DATASET == 'London':
        nest_1 = [1,2] # walk+bike
        nest_2 = [3,4] # auto
    elif Dependent_var == 'MODE' and DATASET == 'SG':
        nest_1 = [1,2] # walk + pt
        nest_2 = [3,4,5] # ride share drive av

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

    accuracy = []


        #######################training
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
        beta_dic[asc_name] = Beta(asc_name, 0, None, None, 0)

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

    MU_motor = Beta('MU_nest1', 1, None, None, 0)
    MU_unmotor = Beta('MU_nest2', 1, None, None, 1)
    N_motor = MU_motor, nest_1
    N_unmotor = MU_unmotor, nest_2

    nests = N_motor, N_unmotor

    logprob = models.lognested(V, av, nests, MODE)
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=n_jobs)
    biogeme.modelName = html_name_cv
    # if '100k' in model_name:
    #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter = 200, method = 'BFGS')
    # else:
    #     betas, beta_name_list = biogeme.estimate_baichuan(max_iter = 200, method='BFGS')
    betas, beta_name_list, write_html_time = biogeme.estimate_baichuan(max_iter=None, method='BFGS',Write_html=True) # max_iter = None means no max_iter used
    write_html_time_total += write_html_time

    beta = {}
    for k in range(len(beta_name_list)):
        beta_name = beta_name_list[k]
        beta[beta_name] = betas[k]

    biogeme_file={'V':V, 'av':av, 'nests':nests}


    ###########prediction
    Results = pd.DataFrame()


    min_X = np.min(X_test[:, col_cost_id])
    max_X = np.max(X_test[:, col_cost_id])

    all_points = np.linspace(min_X, max_X, num=num_points)

    prop_list = []

    for point_value in all_points:
        data_test = data_test_raw.copy()
        data_test.loc[:,col_cost] = point_value

        acc, data_test = predict_NL(beta, biogeme_file, data_test, mode_list, Dependent_var)
        #
        # y_pred = data_test['CHOOSE'].values
        # prop = sum(y_pred == MODE_WANT) / len(y_pred)

        y_prob = data_test.loc[:,'prob_MODE_' + str(MODE_WANT)].values
        prop = np.mean(y_prob)

        prop_list.append(prop)

        #
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

    Results = Results.append(
        pd.DataFrame({'Model': [model_name] * num_points, 'x_values': all_points, 'pred_prop': prop_list}))

    Results.to_csv(output_file_path, index=False)

if __name__ == '__main__':

    tic = time.time()
    # Parameters:

    # sample_size_list = ['1k','10k','100k']
    sample_size_list = ['10k']

    computer_info = 'I9-9900K'

    Re_run = True # True: rerun all models, False: if there are results existed, jump it
    n_jobs = 1
    # Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']
    Dependent_var_list = ['MODE']

    Estimator_name = 'NL_biogeme_RandSpec'
    # DATASET_list = ['NHTS', 'London', 'SG']
    DATASET_list = ['London']


    col_cost = 'dur_pt_bus' # 'cost_driving_fuel' #
    num_points = 100

    MODE_WANT = 3 #4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive
    # (1: walk, 2: cycle, 3: public transport, 4:drive)



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
                    output_file_path = 'Demand_curve_results/Results_London_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    data_name_read = 'train_data_London.csv'
                    data_test_read = 'test_data_London.csv'
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

                model_run(data, data_test, col_cost, num_points, MODE_WANT, output_file_path, 5, Re_run, computer_info, Estimator_name, n_jobs, Dependent_var, DATASET)












