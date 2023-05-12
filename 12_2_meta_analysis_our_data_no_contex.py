import pandas as pd
import numpy as np
import sys

sys.path.insert(0, 'edited_biogeme/')
import database as db
import biogeme as bio
from expressions import *
from sklearn.model_selection import KFold
import time
import os
import statsmodels.api as sm

def get_data_info():
    res = {}
    sg = pd.read_csv('SG_dataset/data_SG_mode_choice_1k.csv')
    input_dim = len(set(sg.columns).difference({'MODE'}))
    num_alt = len(set(sg['MODE']))
    res[('SG', 'MODE')] = {'input_dim': input_dim, 'num_of_alternative': num_alt, 'task': 'TM', 'location': 'AS'}
    ##
    london = pd.read_csv('London_dataset/data_London_mode_choice_1k.csv')
    input_dim = len(set(london.columns).difference({'MODE'}))
    num_alt = len(set(london['MODE']))
    res[('London', 'MODE')] = {'input_dim': input_dim, 'num_of_alternative': num_alt, 'task': 'TM', 'location': 'EU'}
    ##
    data = pd.read_csv('data/data_car_ownership_1k.csv')
    input_dim = len(set(data.columns).difference({'CAR_OWN'}))
    num_alt = len(set(data['CAR_OWN']))
    res[('NHTS', 'CAR_OWN')] = {'input_dim': input_dim, 'num_of_alternative': num_alt, 'task': 'AO', 'location': 'US'}

    ##
    data = pd.read_csv('data/data_mode_choice_1k.csv')
    input_dim = len(set(data.columns).difference({'MODE'}))
    num_alt = len(set(data['MODE']))
    res[('NHTS', 'MODE')] = {'input_dim': input_dim, 'num_of_alternative': num_alt, 'task': 'TM', 'location': 'US'}

    ##
    data = pd.read_csv('data/data_trip_purpose_1k.csv')
    input_dim = len(set(data.columns).difference({'TRIPPURP'}))
    num_alt = len(set(data['TRIPPURP']))
    res[('NHTS', 'TRIPPURP')] = {'input_dim': input_dim, 'num_of_alternative': num_alt, 'task': 'TP', 'location': 'US'}

    return res


def process_exp_data():
    data = pd.read_csv("output/All_results.csv")
    res = get_data_info()
    key_for_direct_info = ['input_dim', 'num_of_alternative', 'task', 'location']
    for key in res:
        dataset, depend_var = key
        for var in key_for_direct_info:
            if var not in set(data.columns):
                data[var] = -1

            data.loc[(data['Data_set'] == dataset) & (data['Dependent_var'] == depend_var), var] = res[key][var]

    data['sample_size'] = -1
    data.loc[data['Sample_size'] == '1k', 'sample_size'] = 1000
    data.loc[data['Sample_size'] == '10k', 'sample_size'] = 10000
    data.loc[data['Sample_size'] == '100k', 'sample_size'] = 100000

    data['model'] = data['Clf_cate']
    data['model_pred_accuracy'] = data['Accuracy']
    data.loc[data['model'] == 'BY','model'] = 'BM'
    data.loc[data['model'] == 'Bagging','model'] = 'BAG'
    data.loc[data['model'] == 'Boosting', 'model'] = 'BOOST'
    # filtering
    used_data = data.copy()
    used_data = used_data.loc[used_data['model_pred_accuracy'] > 0]
    used_data = used_data.loc[used_data['num_of_alternative'] > 0]
    used_data = used_data.loc[used_data['input_dim'] > 0]
    used_data = used_data.loc[used_data['sample_size'] > 0]
    return used_data


def process_data_to_pair(data, sample_rate, sample_num):
    used_data = data.loc[:,
                ['model', 'model_pred_accuracy', 'task', 'num_of_alternative', 'input_dim',
                 'sample_size', 'location']].copy()

    # filtering
    used_data = used_data.loc[used_data['model_pred_accuracy'] > 0]
    used_data = used_data.loc[used_data['num_of_alternative'] > 0]
    used_data = used_data.loc[used_data['input_dim'] > 0]
    used_data = used_data.loc[used_data['sample_size'] > 0]
    used_data = used_data.loc[~used_data['model'].isin(['RBM', 'OTH'])]

    #

    used_data['model_id'] = list(range(len(used_data)))
    used_data['key'] = 1
    used_data_cross = used_data.merge(used_data, on=['key'])
    used_data_cross = used_data_cross.sample(frac=1, random_state=11)
    used_data_cross = used_data_cross.loc[used_data_cross['model_id_x'] <= used_data_cross['model_id_y']].reset_index(
        drop=True) # use unique pairs

    used_data_cross['left_model_better'] = 0
    used_data_cross.loc[
        used_data_cross['model_pred_accuracy_x'] >= used_data_cross['model_pred_accuracy_y'], 'left_model_better'] = 1
    print('num_sample', len(used_data_cross))
    print("positive sample", sum(used_data_cross['left_model_better']))
    # features
    used_data_cross['model_family_x'] = used_data_cross['model_x']
    used_data_cross['model_family_y'] = used_data_cross['model_y']
    used_data_cross.loc[used_data_cross['model_x'].isin(['MNL', 'NL', 'MXL', 'FL']), 'model_family_x'] = 'DCM'
    used_data_cross.loc[used_data_cross['model_y'].isin(['MNL', 'NL', 'MXL', 'FL']), 'model_family_y'] = 'DCM'
    all_model_family = list(set(used_data_cross['model_family_x']).union(set(used_data_cross['model_family_y'])))

    print('all_model_family_num', len(all_model_family))
    # create col
    base_model_family = 'DCM'
    for key in all_model_family:
        if key == base_model_family:
            continue
        col_name = 'if_M_' + key + '_x'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['model_family_x'] == key, col_name] = 1
        col_name = 'if_M_' + key + '_y'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['model_family_y'] == key, col_name] = 1

    all_tasks = list(set(data['task']))
    for key in all_tasks:
        col_name = 'if_T_' + key + '_x'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['task_x'] == key, col_name] = 1
        col_name = 'if_T_' + key + '_y'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['task_y'] == key, col_name] = 1

    all_location = list(set(data['location']))
    for key in all_location:
        col_name = 'if_L_' + key + '_x'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['location_x'] == key, col_name] = 1
        col_name = 'if_L_' + key + '_y'
        used_data_cross[col_name] = 0
        used_data_cross.loc[used_data_cross['location_y'] == key, col_name] = 1

    # data too large, only consider part of them
    if sample_rate:
        used_data_cross = used_data_cross.sample(frac=sample_rate, random_state=123)
    elif sample_num:
        used_data_cross = used_data_cross.sample(n=sample_num, random_state=123)

    print('data size', len(used_data_cross))
    return used_data_cross, all_tasks, all_location, all_model_family


def estimate(data, all_tasks, all_location, all_model_family, save_name):
    data['sample_size_x'] /= 1000  # scale for fast convergence
    data['sample_size_y'] /= 1000

    n_jobs = 4
    mode_list = ['Left', 'Right']
    Dependent_var = 'left_model_better'
    all_av_var = []
    for ele in mode_list:
        av_name = ele + '_' + 'AV'
        data[av_name] = 1
        all_av_var.append(av_name)

    x_columns_both_modes = []
    # for key in all_tasks:
    #     col_name = 'if_T_' + key
    #     x_columns_both_modes.append(col_name)
    # for key in all_location:
    #     col_name = 'if_L_' + key
    #     x_columns_both_modes.append(col_name)
    base_model_family = 'DCM'
    for key in all_model_family:
        if key == base_model_family:
            continue
        col_name = 'if_M_' + key
        x_columns_both_modes.append(col_name)

    numerical_col = []#['num_of_alternative', 'input_dim', 'sample_size']

    used_col_data = []
    for key in x_columns_both_modes + numerical_col:
        used_col_data.append(key + '_x')
        used_col_data.append(key + '_y')

    data_used = data.loc[:, used_col_data + [Dependent_var] + all_av_var]
    database = db.Database("NHTS", data_used)

    # assume all available
    base_mode = 'Left'
    beta_dic = {}
    for mode in mode_list:
        asc_name = 'B___' + 'ASC' + '___' + mode
        if mode == base_mode:
            beta_dic[asc_name] = Beta(asc_name, 0, None, None, 1)
        else:
            beta_dic[asc_name] = Beta(asc_name, 0, None, None, 0)

    for name in x_columns_both_modes:
        beta_name = 'B___' + name
        beta_dic[beta_name] = Beta(beta_name, 0, None, None, 0)

    for name in numerical_col:
        beta_name = 'B___' + name
        beta_dic[beta_name] = Beta(beta_name, 0, None, None, 0)

    av_variable_dic = {}
    for ele in mode_list:
        av_name = ele + '_' + 'AV'
        av_variable_dic[av_name] = Variable(av_name)
    MODE = Variable(Dependent_var)

    U = {}
    for mode in mode_list:
        asc_name = 'B___' + 'ASC' + '___' + mode
        U[mode] = beta_dic[asc_name]
        for feature in x_columns_both_modes:
            beta_name = 'B___' + feature
            if mode == 'Left':
                var_name = feature + '_x'
            elif mode == 'Right':
                var_name = feature + '_y'
            else:
                print('error')
                exit()
            Var = Variable(var_name)
            U[mode] += beta_dic[beta_name] * Var
        for feature in numerical_col:
            beta_name = 'B___' + feature
            if mode == 'Left':
                var_name = feature + '_x'
            elif mode == 'Right':
                var_name = feature + '_y'
            else:
                print('error')
                exit()
            Var = Variable(var_name)
            U[mode] += beta_dic[beta_name] * Var

    V = {}
    for choice in mode_list:
        if choice == 'Left':
            idx = 1
        elif choice == 'Right':
            idx = 0
        else:
            print('Error')
            exit()
        V[idx] = U[choice]

    av = {}
    for choice in mode_list:
        if choice == 'Left':
            idx = 1
        elif choice == 'Right':
            idx = 0
        else:
            print('Error')
            exit()
        av_name = choice + '_' + 'AV'
        av[idx] = av_variable_dic[av_name]

    logprob = bioLogLogit(V, av, MODE)
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=n_jobs)
    html_path = 'html_meta_analysis/'
    biogeme.modelName = html_path + save_name
    print('start biogeme estimate...')
    results = biogeme.estimate()
    results.writeLaTeX()


def linear_regression(data, save_name):

    feature_col = [] # ['num_of_alternative', 'input_dim', 'sample_size']

    all_model_family = set(data['model'])
    used_data = data.copy()
    # create col
    base_model_family = 'DCM'
    for key in all_model_family:
        if key == base_model_family:
            continue
        col_name = 'if_M_' + key
        used_data[col_name] = 0
        used_data.loc[used_data['model'] == key, col_name] = 1
        feature_col.append(col_name)
    #
    # all_tasks = list(set(data['task']))
    # for key in all_tasks:
    #     col_name = 'if_T_' + key
    #     used_data[col_name] = 0
    #     used_data.loc[used_data['task'] == key, col_name] = 1
    #     feature_col.append(col_name)
    #
    # all_location = list(set(data['location']))
    # for key in all_location:
    #     col_name = 'if_L_' + key
    #     used_data[col_name] = 0
    #     used_data.loc[used_data['location'] == key, col_name] = 1
    #     feature_col.append(col_name)

    used_data['constant'] = 1
    feature_col.append('constant')
    y_col = 'model_pred_accuracy'
    mod = sm.OLS(used_data[y_col], used_data.loc[:, feature_col])
    res = mod.fit()
    stat_table = res.summary2().tables[0]
    restable  = res.summary2().tables[1]
    # print(type(restable))
    restable.to_csv('html_meta_analysis/' + save_name + '.csv')
    stat_table.to_csv('html_meta_analysis/' + save_name + 'stats_table.csv')


def output_res_to_table_DCM(save_name):
    digit_num = 3
    ASC = ['B___ASC___Right']
    Model_dummy = ["B___if_M_BAG", "B___if_M_BM" , "B___if_M_BOOST" , "B___if_M_DA", "B___if_M_DNN", "B___if_M_DT", "B___if_M_GLM","B___if_M_GP", "B___if_M_KNN" , "B___if_M_RF",
                   "B___if_M_SVM"]
    loc_dummy = [  "B___if_L_AS",   "B___if_L_AM", "B___if_L_EU"]
    task_dummy = ["B___if_T_AO", "B___if_T_TP"]
    other = ["B___input_dim", "B___num_of_alternative", "B___sample_size"]
    para_info = {}
    all_var_seq = ASC + Model_dummy + loc_dummy + task_dummy + other
    html_path = 'html_meta_analysis/'
    file_name = html_path + save_name + '.tex'
    flag = 0
    with open(file_name) as fin:
        for line in fin:
            if '\\section{Parameter estimates}' in line:
                flag = 1
            if '\\end{tabular}' in line and flag == 1:
                flag = 0
                print(line)
                break
            if flag == 1:
                for var in all_var_seq:
                    var_in_tex = var.replace('_','\\_')
                    if var_in_tex in line:

                        info = line.split('&')
                        var_name = info[0].replace(' ','')
                        if var_in_tex != var_name:
                            continue
                        para = info[1].replace(' ','')
                        std_error = info[2].replace(' ','')
                        t_test = info[3].replace(' ','')
                        t_test = t_test.replace('-','')
                        p_value = info[4].replace(' ','')
                        para_info[var] = [para,std_error, p_value]

    data = {'var':[],'value_std_er':[]}
    for key in all_var_seq:
        if key in para_info:
            if 'sigma_s_tidle' in key:
                SIGMA_FIX_TO_BE_POSITIVE = True
            else:
                SIGMA_FIX_TO_BE_POSITIVE = False
            data['var'].append(key)
            if 'e' in para_info[key][2]:
                part1 = float(para_info[key][2].split('e')[0])
                part2 = float(para_info[key][2].split('e')[1])
                p_value_float = part1 * (pow(10,part2))
            else:
                p_value_float = float(para_info[key][2])
            if p_value_float> 0.05: #p_value
                star_str = ''
            if p_value_float <= 0.05: #p_value
                star_str = ' *'
            if p_value_float <= 0.01: #p_value
                star_str = ' **'
            if p_value_float <= 0.001: #p_value
                star_str = ' ***'

            save_value_est = round(float(para_info[key][0]),digit_num)
            save_value_std =  round(float(para_info[key][1]),digit_num)

            def process_num(save_value):
                str_ = str(save_value)
                flag_minus = False
                if '-' in str_:
                    str_ = str_.replace('-','')
                    flag_minus = True

                if len(str_) == 5: # 0.000
                    if flag_minus:
                        str_ = '-' + str_
                    if SIGMA_FIX_TO_BE_POSITIVE:
                        str_ = str_.replace('-','')
                    return str_
                elif len(str_) < 5: # need add zero
                    num_zero = 5-len(str_)
                    if flag_minus:
                        str_ = '-' + str_ + ''.join(['0'] * num_zero)
                    else:
                        str_ = str_ + ''.join(['0'] * num_zero)
                    if SIGMA_FIX_TO_BE_POSITIVE:
                        str_ = str_.replace('-','')
                    return str_
                else:
                    print('error')
                    print(str_)
                    exit()

            save_value_est = process_num(save_value_est)
            save_value_std = process_num(save_value_std)

            data['value_std_er'].append(str(save_value_est) + ' ' + '(' + str(save_value_std) + ')' + star_str)

    data = pd.DataFrame(data)
    output_name = file_name.replace('.tex','.csv')
    data.to_csv(output_name,index=False)



def output_res_to_table_LR(save_name):
    digit_num = 3
    para_info = {}

    html_path = 'html_meta_analysis/'
    file_name = html_path + save_name + '.csv'
    res = pd.read_csv(file_name)


    ASC = ['constant']
    Model_dummy = ["if_M_BAG", "if_M_BM" , "if_M_BOOST" , "if_M_DA", "if_M_DNN", "if_M_DT", "if_M_GLM","if_M_GP", "if_M_KNN" , "if_M_RF",
                   "if_M_SVM"]
    loc_dummy = [  "if_L_AS",   "if_L_AM", "if_L_EU"]
    task_dummy = ["if_T_AO", "if_T_TP"]
    other = ["input_dim", "num_of_alternative", "sample_size"]
    all_var_seq = ASC + Model_dummy + loc_dummy + task_dummy + other
    for var, para, std_error, p_value in zip(res.iloc[:,0], res.iloc[:,1],res.iloc[:,2],res.iloc[:,4]):
        para_info[var] = [para, std_error, p_value]

    data = {'var':[],'value_std_er':[]}
    for key in all_var_seq:
        if key in para_info:
            if 'sigma_s_tidle' in key:
                SIGMA_FIX_TO_BE_POSITIVE = True
            else:
                SIGMA_FIX_TO_BE_POSITIVE = False
            data['var'].append(key)

            p_value_float = float(para_info[key][2])
            if p_value_float> 0.05: #p_value
                star_str = ''
            if p_value_float <= 0.05: #p_value
                star_str = ' *'
            if p_value_float <= 0.01: #p_value
                star_str = ' **'
            if p_value_float <= 0.001: #p_value
                star_str = ' ***'

            save_value_est = round(float(para_info[key][0]),digit_num)
            save_value_std =  round(float(para_info[key][1]),digit_num)

            def process_num(save_value):
                str_ = str(save_value)
                flag_minus = False
                if '-' in str_:
                    str_ = str_.replace('-','')
                    flag_minus = True

                if len(str_) == 5: # 0.000
                    if flag_minus:
                        str_ = '-' + str_
                    if SIGMA_FIX_TO_BE_POSITIVE:
                        str_ = str_.replace('-','')
                    return str_
                elif len(str_) < 5: # need add zero
                    num_zero = 5-len(str_)
                    if flag_minus:
                        str_ = '-' + str_ + ''.join(['0'] * num_zero)
                    else:
                        str_ = str_ + ''.join(['0'] * num_zero)
                    if SIGMA_FIX_TO_BE_POSITIVE:
                        str_ = str_.replace('-','')
                    return str_
                else:
                    print('error')
                    print(str_)
                    exit()
            save_value_est = process_num(save_value_est)
            save_value_std = process_num(save_value_std)

            data['value_std_er'].append(str(save_value_est) + ' ' + '(' + str(save_value_std) + ')' + star_str)
        else:
            data['var'].append(key)
            data['value_std_er'].append('NA')

    data = pd.DataFrame(data)
    output_name = file_name.replace('.csv','_para.csv')
    data.to_csv(output_name,index=False)

if __name__ == '__main__':
    data = process_exp_data()
    DCM = False
    LR = True

    # DCM
    if DCM:
        ESTIMATE = True
        sample_size = 2000
        save_name = f'meta_analysis_our_data_no_context_{sample_size}'
        if ESTIMATE:
            data, all_tasks, all_location, all_model_family = process_data_to_pair(data, sample_rate=None, sample_num = sample_size)
            estimate(data, all_tasks, all_location, all_model_family, save_name=save_name)
        output_res_to_table_DCM(save_name)

    # linear regression
    if LR:
        ESTIMATE = True
        save_name = f'meta_analysis_our_data_no_context_LR'
        if ESTIMATE:
            linear_regression(data, save_name)
        output_res_to_table_LR(save_name)
