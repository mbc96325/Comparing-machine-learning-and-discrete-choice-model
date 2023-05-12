import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel, SelectKBest, f_classif

London_data = pd.read_csv('London_dataset/data_London_mode_choice_100k.csv')

np.random.seed(1234)
London_data = London_data.sample(frac = 1)
data = London_data.iloc[0:20000]



# col_name = 'cost_driving_fuel'
# train_max_row = train_data.loc[train_data[col_name] == np.max(train_data[col_name])]
# train_min_row = train_data.loc[train_data[col_name] == np.min(train_data[col_name])]
################
ref_table = pd.read_csv('London_dataset/Input_variables_London.csv')
x_columns = list(ref_table.loc[ref_table['Mode_choice_input'] == 1, 'Input_variables'])

print('total_var', len(x_columns))

normalized_vec = list(ref_table.loc[ref_table['Normalize'] == 1, 'Input_variables'])
X = np.array(data.loc[:, normalized_vec])
X_new = MinMaxScaler().fit_transform(X)  # normalize X
data.loc[:, normalized_vec] = X_new
################
train_data = data.iloc[0:10000]
test_data = data.iloc[10000:20000]

print('train size', len(train_data))
print('test size', len(test_data))

train_data.to_csv('London_dataset/train_data_London.csv',index=False)
test_data.to_csv('London_dataset/test_data_London.csv',index=False)


###############################
London_data = pd.read_csv('London_dataset/data_London_mode_choice_100k.csv')

np.random.seed(5678)
London_data = London_data.sample(frac = 1)
data = London_data.iloc[0:2000]


################
ref_table = pd.read_csv('London_dataset/Input_variables_London.csv')
x_columns = list(ref_table.loc[ref_table['Mode_choice_input'] == 1, 'Input_variables'])

print('total_var', len(x_columns))

normalized_vec = list(ref_table.loc[ref_table['Normalize'] == 1, 'Input_variables'])
X = np.array(data.loc[:, normalized_vec])
X_new = MinMaxScaler().fit_transform(X)  # normalize X
data.loc[:, normalized_vec] = X_new
################
train_data_1k = data.iloc[0:1000]
test_data_1k = data.iloc[1000:2000]
print('train size', len(train_data_1k))
print('test size', len(test_data_1k))



train_data_1k.to_csv('London_dataset/train_data_London_1k.csv',index=False)
test_data_1k.to_csv('London_dataset/test_data_London_1k.csv',index=False)

a=1

