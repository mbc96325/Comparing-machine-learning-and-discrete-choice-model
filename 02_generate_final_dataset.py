import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel, SelectKBest, f_classif
import sys
import pickle

def generate_mode_choice_ds(data, ref_table):
    data['MODE'] = 6 # others
    # 1 walk and bike 2 car 3 suv 4 van and truck 5 public transit, 6 other
    columns = [[1,2],[3],[4],[5,6],[10,11,12,13,14,15,16]]
    for i in range(len(columns)):
        data.loc[data['TRPTRANS'].isin(columns[i]),'MODE'] = i+1
    x_columns = list(ref_table.loc[ref_table['Mode_choice_input'] == 1, 'Input_variables'])
    normalized_vec = list(ref_table.loc[ref_table['Normalize'] == 1, 'Input_variables'])
    X = np.array(data.loc[:,normalized_vec])
    # X_new = MinMaxScaler().fit_transform(X)  # normalize X
    X_new = StandardScaler().fit_transform(X)  # normalize X
    data.loc[:,normalized_vec] = X_new

    # feature select
    K_best = 50
    Feature_select = SelectKBest(f_classif, k=K_best).fit(data.loc[:,x_columns], data.loc[:,'MODE'])
    used_feature_index = Feature_select.get_support()
    used_feature = [x_columns[i] for i in range(len(used_feature_index)) if used_feature_index[i]]

    data_final = data.loc[:,['MODE'] + used_feature]
    file_name = {'1k': (1000, 4), '10k': (10000, 5), '100k': (100000, 6)}
    for idx in file_name:
        N_sample, rand_seed_ = file_name[idx]
        df_elements = data_final.sample(n=N_sample, random_state = rand_seed_ ) # test data for coding
        df_elements.to_csv('data/data_mode_choice_'+ idx +'.csv', index=False)

def generate_car_ownership(data,ref_table):
    data['CAR_OWN'] = -1 # others
    # 1 carown = 0;
    # 2 carown = 1 ...
    # 3 carown = 2 ...
    # 4 carown = 3 ...
    # 5: carown>=4
    value = [0,1,2,3]
    count = 1
    for val in value:
        data.loc[data['HHVEHCNT']==val,'CAR_OWN'] = count
        count += 1
    data.loc[data['HHVEHCNT'] >= 4, 'CAR_OWN'] = count
    data = data.loc[data['CAR_OWN']!= -1]
    print(data.loc[:,'CAR_OWN'].value_counts())
    x_columns = list(ref_table.loc[ref_table['Car_ownership_input'] == 1, 'Input_variables'])
    print(len(data))
    data = data.loc[:,['CAR_OWN','HOUSEID', 'PERSONID'] + x_columns]
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data_group_count = data.groupby(['HOUSEID','PERSONID'])['CAR_OWN'].count().reset_index()
    if sum(data_group_count['CAR_OWN']) == len(data):
        print('no duplicate individuals')
    else:
        print('duplicate individual, please check input var')
        data_group_count_ERROR = data_group_count.loc[data_group_count['CAR_OWN'] > 1]
        data_test = data.drop_duplicates(['HOUSEID','PERSONID'])
        print(len(data))

    X = np.array(data.loc[:,x_columns])
    X_new = MinMaxScaler().fit_transform(X)  # normalize X
    data.loc[:,x_columns] = X_new

    # feature select
    K_best = 50
    Feature_select = SelectKBest(chi2, k=K_best).fit(data.loc[:,x_columns], data.loc[:,'CAR_OWN'])
    used_feature_index = Feature_select.get_support()
    used_feature = [x_columns[i] for i in range(len(used_feature_index)) if used_feature_index[i]]
    #---------
    data_final = data.loc[:,['CAR_OWN'] + used_feature]
    file_name = {'1k':(1000,7),'10k':(10000,8),'100k':(100000,9)} #(Sample size, random seed)
    for idx in file_name:
        N_sample, rand_seed_ = file_name[idx]
        df_elements = data_final.sample(n=N_sample, random_state = rand_seed_ ) # test data for coding
        df_elements.to_csv('data/data_car_ownership_'+ idx +'.csv', index=False)

def generate_trip_purpose_ds(data, ref_table):
    data['TRIPPURP'] = -1 # others
    # 1 Home based other 2 home based shop 3 home based social 4 home based work 5 non home based
    columns = ['TRIPPURP_HBO','TRIPPURP_HBSHOP','TRIPPURP_HBSOCREC','TRIPPURP_HBW','TRIPPURP_NHB']
    count = 1
    for name in columns:
        data.loc[data[name]==1,'TRIPPURP'] = count
        count += 1
    data = data.loc[data['TRIPPURP']!= -1]
    x_columns = list(ref_table.loc[ref_table['Trip_purpose_input'] == 1, 'Input_variables'])
    X = np.array(data.loc[:,x_columns])
    X_new = MinMaxScaler().fit_transform(X)  # normalize X
    data.loc[:,x_columns] = X_new

    # feature select
    K_best = 50
    Feature_select = SelectKBest(chi2, k=K_best).fit(data.loc[:,x_columns], data.loc[:,'TRIPPURP'])
    used_feature_index = Feature_select.get_support()
    used_feature = [x_columns[i] for i in range(len(used_feature_index)) if used_feature_index[i]]
    #---------
    data_final = data.loc[:,['TRIPPURP'] + used_feature]
    file_name = {'1k':(1000,1),'10k':(10000,2),'100k':(100000,3)}
    for idx in file_name:
        N_sample, rand_seed_ = file_name[idx]
        df_elements = data_final.sample(n=N_sample, random_state = rand_seed_ ) # test data for coding
        df_elements.to_csv('data/data_trip_purpose_'+ idx +'.csv', index=False)

def process_NHTS():
    tic = time.time()
    data = pd.read_csv('data/data_input_V1.csv')
    ref_table = pd.read_csv('data/Input_variables.csv')
    print('Read raw data time:', round(time.time() - tic, 1),'s')
    # generate_mode_choice_ds(data, ref_table)
    # generate_trip_purpose_ds(data, ref_table)
    generate_car_ownership(data, ref_table)


def generate_mode_choice_London(data, ref_table):
    print('total number of samples', len(data))

    data['MODE'] = data['travel_mode']
    # (1: walk, 2: cycle, 3: public transport, 4:drive)
    x_columns = list(ref_table.loc[ref_table['Mode_choice_input'] == 1, 'Input_variables'])

    print('total_var', len(x_columns))

    normalized_vec = list(ref_table.loc[ref_table['Normalize'] == 1, 'Input_variables'])
    X = np.array(data.loc[:,normalized_vec])
    X_new = MinMaxScaler().fit_transform(X)  # normalize X
    data.loc[:,normalized_vec] = X_new


    # feature select
    K_best = 40
    Feature_select = SelectKBest(chi2, k=K_best).fit(data.loc[:,x_columns], data.loc[:,'MODE'])
    used_feature_index = Feature_select.get_support()
    used_feature = [x_columns[i] for i in range(len(used_feature_index)) if used_feature_index[i]]

    data_final = data.loc[:,['MODE'] + used_feature]
    file_name = {'1k': (1000, 10), '10k': (10000, 11), '100k': (80000, 12)}
    for idx in file_name:
        N_sample, rand_seed_ = file_name[idx]
        df_elements = data_final.sample(n=N_sample, random_state = rand_seed_ ) # test data for coding
        df_elements.to_csv('London_dataset/data_London_mode_choice_'+ idx +'.csv', index=False)

def process_London():
    tic = time.time()
    data = pd.read_csv('London_dataset/data_input_London.csv')
    ref_table = pd.read_csv('London_dataset/Input_variables_London.csv')
    print('Read raw data time:', round(time.time() - tic, 1),'s')
    generate_mode_choice_London(data, ref_table)


def generate_mode_choice_SG(data, ref_table):
    print('total number of samples', len(data))

    data['MODE'] = data['choice'] + 1
    # key_choice_index = {'Walk': 1, 'PT': 2, 'RH': 3, 'AV': 5, 'Drive': 4}
    x_columns = list(ref_table.loc[ref_table['Mode_choice_input'] == 1, 'Input_variables'])

    print('total_var', len(x_columns))

    normalized_vec = list(ref_table.loc[ref_table['Normalize'] == 1, 'Input_variables'])
    X = np.array(data.loc[:,normalized_vec])
    X_new = StandardScaler().fit_transform(X)  # normalize X
    data.loc[:,normalized_vec] = X_new


    # no feature select
    # K_best = len(x_columns)
    # Feature_select = SelectKBest(chi2, k=K_best).fit(data.loc[:,x_columns], data.loc[:,'MODE'])
    # used_feature_index = Feature_select.get_support()
    # used_feature = [x_columns[i] for i in range(len(used_feature_index)) if used_feature_index[i]]
    used_feature = x_columns
    data_final = data.loc[:,['MODE'] + used_feature]
    file_name = {'1k': (1000, 13), '10k': (8000, 14)}
    for idx in file_name:
        N_sample, rand_seed_ = file_name[idx]
        df_elements = data_final.sample(n=N_sample, random_state = rand_seed_ ) # test data for coding
        df_elements.to_csv('SG_dataset/data_SG_mode_choice_'+ idx +'.csv', index=False)

def process_SG():
    tic = time.time()
    data = pd.read_csv('SG_dataset/data_AV_Singapore_v1_sp_full_nonstand.csv')
    ref_table = pd.read_csv('SG_dataset/Input_variables_SG.csv')
    print('Read raw data time:', round(time.time() - tic, 1),'s')
    generate_mode_choice_SG(data, ref_table)

if __name__ == '__main__':

    # process_NHTS()
    process_London()
    #process_SG()
    # a=1