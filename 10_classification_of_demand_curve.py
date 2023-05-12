def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn # what a elegant way to avoid warnings!


import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def get_features(curve_file, output_file_path,sample_size,dataset,dependent_var):
    data_points = {'Model':[],'file_name':[],'num_increase':[],'num_decrease':[],'no_change':[],'Sample_size':[],'Data_set':[],'Dependent_var':[]}
    for model_name, group in curve_file.groupby(['Model']):
        group_to_anal = group.copy()
        group_to_anal['next_point'] = group_to_anal['pred_prop'].shift(-1)
        group_to_anal['diff'] = group_to_anal['next_point'] - group_to_anal['pred_prop']
        num_increase = sum(group_to_anal['diff'] > 0)
        num_decrease =  sum(group_to_anal['diff'] < 0)
        no_change = sum(group_to_anal['diff'] == 0)
        data_points['Model'].append(model_name)
        data_points['num_increase'].append(num_increase)
        data_points['num_decrease'].append(num_decrease)
        data_points['no_change'].append(no_change)
        data_points['file_name'].append(output_file_path)
        data_points['Sample_size'].append(sample_size)
        data_points['Data_set'].append(dataset)
        data_points['Dependent_var'].append(dependent_var)

    return pd.DataFrame(data_points)

def process_curve_features(col_cost, MODE_WANT):
    Results_all = []

    sample_size_list = ['1k', '10k']
    Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']

    DATASET_list = ['London']

    DNN_estimator = ['DNN_AdaBoost','DNN','DNN_GradientBoost']
    dnn_struc_list = [(1,30),(3,30),(5,30),
                      (1,100),(3,100),(5,100),
                      (1,200),(3,200),(5,200)]
    DNN_estimator_name = []
    for dnn_struc in dnn_struc_list:
        numUnits = dnn_struc[1]
        numHlayers = dnn_struc[0]
        DNN_estimator_name += [ele + '_' + str(numUnits) + '_' + str(numHlayers) + '_python' for ele in DNN_estimator]
    Estimator_name_list = ['BY_python','DA_python','DecisionTree_python','Ensemble_python','GenLinear_python','GP_python',
                           'KNN_python','MixL_biogeme_ASUSpec','MNL_biogeme_ASUSpec','NL_biogeme_ASUSpec','SVM_python','BY_R','DA_R','DecisionTree_R',
                           'Ensemble_R','NN_R','Rule_R','SVM_R','Weka_python']
    Estimator_name_list += DNN_estimator_name

    for Estimator_name in Estimator_name_list:
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
                        output_file_path = output_file_path.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
                            int(MODE_WANT)) + '.csv')
                        if os.path.exists(output_file_path):
                            Results = pd.read_csv(output_file_path)
                            result_temp = get_features(Results, output_file_path, sample_size, DATASET,Dependent_var)
                            Results_all.append(result_temp)
                            # a=1
                            # Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                        else:
                            if sample_size == '1k':
                                if 'GP' in Estimator_name or 'MixL' in  Estimator_name or "SVM_R" in Estimator_name:
                                    print(output_file_path, 'does not exist')
                                else:
                                    continue
                            if ('GP' in Estimator_name or 'MixL' in Estimator_name) and sample_size == '10k':
                                continue
                            print(output_file_path, 'does not exist')
                    # elif DATASET == 'SG':
                    #     output_file_path = 'Results/Results_SG_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    #     if os.path.exists(output_file_path):
                    #         Results = pd.read_csv(output_file_path)
                    #         Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                    #     else:
                    #         print(output_file_path, 'does not exist')
                    # else:
                    #     output_file_path = 'Results/Results_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                    #     if os.path.exists(output_file_path):
                    #         Results = pd.read_csv(output_file_path)
                    #         Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                    #     else:
                    #         print(output_file_path, 'does not exist')

    Results_all_df = pd.concat(Results_all)


    return Results_all_df



def plot_demand_curve(data_label, label_cluster, save_fig, cluster_label, col_cost, MODE_WANT):

    colors = sns.color_palette()
    font_size = 16
    fig, ax = plt.subplots(figsize=(8, 6))
    y_mean = []

    for model_name, file_path in zip(data_label['Model'], data_label['file_name']):
        data_read = pd.read_csv(file_path)
        data_plot = data_read.loc[data_read['Model'] == model_name]
        x_plot = np.array(data_plot['x_values'])
        y_plot = np.array(data_plot['pred_prop'])

        y_mean.append(y_plot)

        plt.plot(x_plot, y_plot, color=colors[0], alpha = 0.25, linewidth = 1)
        # plt.scatter(Results_all_single['Model'], Results_all_single['avg_acc'],color = 'red', label ='Mean',zorder = 5)

    y_mean_np = np.array(y_mean)
    y_mean_plot = np.mean(y_mean_np,axis = 0)

    plt.plot(x_plot, y_mean_plot, color=colors[0], linewidth = 2, label = 'Mean')
    plt.legend(fontsize = font_size)

    pos_x = int(np.round(len(x_plot)/3))

    label_cluster_to_text = {}
    for label, name in zip(label_cluster['cluster_label'], label_cluster['Label_name']):
        label_cluster_to_text[label] = name


    if isinstance(cluster_label, int):
        text_to_show = 'Cluster ' + str(cluster_label) + ': ' + label_cluster_to_text[cluster_label]
    else:
        text_to_show = cluster_label
    plt.text(x_plot[pos_x], 0.9, text_to_show, size=font_size*1.1,
             ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )
    text_to_show2 = '# models = ' + str(len(data_label))
    plt.text(x_plot[pos_x], 0.8, text_to_show2, size=font_size*1.1,
             ha="center", va="center")

    if MODE_WANT == 4:
        plt.xlabel('Normalized driving cost',fontsize=font_size)
        plt.ylabel('Probability of driving',fontsize=font_size)
    elif MODE_WANT == 3:
        plt.xlabel('Normalized PT time',fontsize=font_size)
        plt.ylabel('Probability of PT',fontsize=font_size)
    plt.ylim([0,1])
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()

    img_output = 'img/demand_curve_all_' + str(cluster_label) + '.jpg'
    img_output = img_output.replace('.jpg', '_' + col_cost + '_' + 'mode' + str(
        int(MODE_WANT)) + '.jpg')

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig(img_output, dpi=200)

def classification(Results_all_df, col_cost, MODE_WANT, save_fig):
    # a=1

    feature_lists = ['num_increase','num_decrease','no_change']
    X = np.array(Results_all_df[feature_lists])
    num_cluster = 4
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    labels = kmeans.predict(X)

    Results_all_df['cluster_label'] = labels + 1

    # if MODE_WANT == 4:
    #     label_cluster = {'cluster_label':[1,2,3,4], 'Label_name':['Flat','Decrease','Increase','Other']}
    # elif MODE_WANT == 3:
    #     label_cluster = {'cluster_label':[1,2,3,4], 'Label_name':['Other','Flat','Increase','Decrease']}

    feature_each_cluster = {'cluster_label':[], 'num_increase':[], 'num_decrease':[],'no_change':[]}
    for key in range(num_cluster):
        data_cluster =  Results_all_df.loc[Results_all_df['cluster_label'] == key + 1]
        feature_each_cluster['cluster_label'].append(key + 1)
        feature_each_cluster['num_increase'].append(np.mean(data_cluster['num_increase']))
        feature_each_cluster['num_decrease'].append(np.mean(data_cluster['num_decrease']))
        feature_each_cluster['no_change'].append(np.mean(data_cluster['no_change']))

    feature_each_cluster_df = pd.DataFrame(feature_each_cluster)

    max_feature_dict = {'Flat':'no_change', 'Decrease':'num_decrease', 'Increase':'num_increase'}
    label_cluster = {'cluster_label':[],'Label_name':[]}
    for key in max_feature_dict:
        feature_each_cluster_df_sort = feature_each_cluster_df.sort_values([max_feature_dict[key]],ascending=False)
        cluster = feature_each_cluster_df_sort['cluster_label'].iloc[0]
        label_cluster['cluster_label'].append(cluster)
        label_cluster['Label_name'].append(key)

    remaining_cluster = list(set(Results_all_df['cluster_label']).difference(set(label_cluster['cluster_label'])))
    label_cluster['cluster_label'].append(remaining_cluster[0])
    label_cluster['Label_name'].append('Other')


    all_labels = np.unique(Results_all_df['cluster_label'])
    all_accuracy_results = pd.read_csv('output/All_results.csv')
    Results_all_df = Results_all_df.merge(all_accuracy_results[['Model','Data_set','Sample_size','Dependent_var','Clf_cate']],
                                                              on=['Model', 'Data_set', 'Sample_size', 'Dependent_var'],
                                                              how='left')
    check = Results_all_df.loc[Results_all_df['Clf_cate'].isna()]
    if len(check) > 0:
        Results_all_df['Clf_cate'] = Results_all_df['Clf_cate'].fillna('DCM') # to be revised


    label_cluster_df = pd.DataFrame(label_cluster)
    Results_all_df_save = Results_all_df[['Model','cluster_label','Clf_cate'] + feature_lists].copy()
    Results_all_df_save = Results_all_df_save.merge(label_cluster_df, on = ['cluster_label'])

    data_output = 'output/Cluster_and_model_name.csv'
    data_output = data_output.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
        int(MODE_WANT)) + '.csv')
    Results_all_df_save.to_csv(data_output,index=False)

    map_from_clf_to_cluster = Results_all_df.groupby(['Clf_cate','cluster_label'])['Model'].count().reset_index()

    cluster_label_df = pd.DataFrame({'cluster_label':list(range(num_cluster))})
    cluster_label_df['cluster_label'] += 1
    all_clf = all_accuracy_results[['Clf_cate']].drop_duplicates()
    all_clf['key'] = 1
    cluster_label_df['key'] = 1
    map_from_clf_to_cluster_all = all_clf.merge(cluster_label_df, on = ['key']).drop(columns=['key'])


    map_from_clf_to_cluster_all = map_from_clf_to_cluster_all.merge(map_from_clf_to_cluster, how = 'left', on =['Clf_cate','cluster_label'])
    map_from_clf_to_cluster_all = map_from_clf_to_cluster_all.rename(columns = {'Model':'Num_model'})
    map_from_clf_to_cluster_all['Num_model'] = map_from_clf_to_cluster_all['Num_model'].fillna(0)


    map_from_clf_to_cluster_all = map_from_clf_to_cluster_all.merge(label_cluster_df, on = ['cluster_label'])

    map_from_clf_to_cluster_all = map_from_clf_to_cluster_all.sort_values(['Clf_cate','cluster_label'])

    data_output = 'output/Cluster_and_clf_cate.csv'
    data_output = data_output.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
        int(MODE_WANT)) + '.csv')

    map_from_clf_to_cluster_all.to_csv(data_output,index=False)

    for key in all_labels:
        data_label = Results_all_df.loc[Results_all_df['cluster_label'] == key]
        plot_demand_curve(data_label, label_cluster, save_fig = save_fig, cluster_label = int(key), col_cost = col_cost, MODE_WANT = MODE_WANT)


    plot_demand_curve(Results_all_df, label_cluster, save_fig = save_fig, cluster_label = 'All', col_cost = col_cost, MODE_WANT = MODE_WANT)

if __name__ == '__main__':
    # a=1

    col_cost = 'cost_driving_fuel' # 'cost_driving_fuel' 'cost_driving_ccharge' #dur_pt_access
    num_points = 100

    MODE_WANT = 4 #4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive
    # (1: walk, 2: cycle, 3: public transport, 4:drive)
    output_file_path = 'output/demand_curve_features.csv'
    output_file_path = output_file_path.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
        int(MODE_WANT)) + '.csv')

    Results_all_df = process_curve_features(col_cost, MODE_WANT)
    Results_all_df.to_csv(output_file_path,index=False)

    Results_all_df = pd.read_csv(output_file_path)
    classification(Results_all_df, col_cost, MODE_WANT, save_fig = 0)
    a=1