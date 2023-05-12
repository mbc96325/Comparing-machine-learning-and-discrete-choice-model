import pandas as pd
from scipy import stats

col_cost = 'dur_pt_access'  # 'cost_driving_fuel' #
num_points = 100

MODE_WANT = 3  # 4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive

all_accuracy_results = pd.read_csv('output/All_results.csv')

demand_curve_features_file_name = 'output/demand_curve_features.csv'
demand_curve_features_file_name = demand_curve_features_file_name.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
    int(MODE_WANT)) + '.csv')

demand_curve_features = pd.read_csv(demand_curve_features_file_name)

# print('num model before',len(demand_curve_features))
demand_curve_features_merge = demand_curve_features.merge(all_accuracy_results, on = ['Model','Data_set','Sample_size','Dependent_var'], how = 'left')
check = demand_curve_features_merge.loc[demand_curve_features_merge['Accuracy'].isna()]
demand_curve_features_merge = demand_curve_features_merge.dropna()

features = ['num_decrease','num_increase','no_change']
y = demand_curve_features_merge['Accuracy'].values

Results = {'Feature':[],'Corr_with_accuracy':[],'p_value':[]}
for key in features:
    x = demand_curve_features_merge[key].values
    corr, p_value = stats.pearsonr(x, y)
    Results['Feature'].append(key)
    Results['Corr_with_accuracy'].append(corr)
    Results['p_value'].append(p_value)

Results_df = pd.DataFrame(Results)

out_put_name = 'output/correlation_between_demand_curve_and_accuracy.csv'
out_put_name = out_put_name.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
    int(MODE_WANT)) + '.csv')
Results_df.to_csv(out_put_name,index=False)

a=1