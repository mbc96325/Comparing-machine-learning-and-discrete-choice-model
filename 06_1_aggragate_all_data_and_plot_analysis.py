import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

colors = sns.color_palette("muted")


def process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all):
    avg_acc = Results.copy()
    avg_acc = avg_acc.loc[avg_acc['Fold'] == 'Average']
    avg_acc['Clf_cate'] = -1
    avg_acc['Programming'] = -1
    avg_acc['Sample_size'] = sample_size
    avg_acc['Dependent_var'] = Dependent_var
    avg_acc['Data_set'] = DATASET
    #################
    direct_clf = ['BY_','DA_','DecisionTree_','SVM_','KNN_','GP_','Rule_','GenLinear_']
    for clf in direct_clf:
        if clf in Estimator_name:
            if clf == 'DecisionTree_':
                avg_acc['Clf_cate'] = 'DT'
            elif clf == 'Rule_':
                avg_acc['Clf_cate'] = 'DT'
            elif clf == 'GenLinear_':
                avg_acc['Clf_cate'] = 'GLM'
            else:
                avg_acc['Clf_cate'] = clf.replace('_','')
            avg_acc['Programming'] = Estimator_name.replace(clf,'')
    if Estimator_name == 'Ensemble_python':
        avg_acc.loc[avg_acc['Model'] == 'BaggingClassifier','Model'] = 'BaggingClassifier_SVM'
        avg_acc.loc[avg_acc['Model'].str.contains('Bagging'),'Clf_cate'] = 'Bagging'
        avg_acc.loc[avg_acc['Model'] == 'RandomForestClassifier', 'Clf_cate'] = 'RF'
        avg_acc.loc[avg_acc['Model'] == 'ExtraTreesClassifier', 'Clf_cate'] = 'RF'
        avg_acc.loc[avg_acc['Model'] == 'VotingClassifier', 'Clf_cate'] = 'Bagging'
        avg_acc.loc[avg_acc['Model'].str.contains('Boost'), 'Clf_cate'] = 'Boosting'
        avg_acc['Programming'] = 'python'
    elif Estimator_name == 'Ensemble_R':
        avg_acc.loc[avg_acc['Model'].str.contains('Boost'), 'Clf_cate'] = 'Boosting'
        avg_acc['Programming'] = 'R'
    elif 'DNN' in Estimator_name:
        avg_acc['Programming'] = 'python_tensorflow'
        if 'Boost' in Estimator_name and 'DNN' in Estimator_name:
            avg_acc['Clf_cate'] = 'Boosting'
        else:
            avg_acc['Clf_cate'] = 'DNN'
    elif 'Boost' in Estimator_name and 'DNN' in Estimator_name:
        avg_acc['Clf_cate'] = 'Boosting'
        avg_acc['Programming'] = 'python_tensorflow'
    elif Estimator_name == 'NN_R':
        avg_acc['Clf_cate'] = 'DNN'
        avg_acc['Programming'] = 'R'
    elif 'biogeme' in Estimator_name:
        avg_acc['Clf_cate'] = 'DCM'
        avg_acc['Programming'] = 'python_biogeme'
    elif Estimator_name == 'Weka_python':
        if avg_acc['Accuracy'].iloc[0]>1:
            avg_acc['Accuracy'] /= 100
        avg_acc['Programming'] = 'Weka'
        avg_acc.loc[avg_acc['Model'] == 'BayesNet_weka', 'Clf_cate'] = 'BY'
        avg_acc.loc[avg_acc['Model'] == 'NaiveBayes_weka', 'Clf_cate'] = 'BY'
        avg_acc.loc[avg_acc['Model'] == 'MLP_weka', 'Clf_cate'] = 'DNN'
        avg_acc.loc[avg_acc['Model'] == 'Logistic_weka', 'Clf_cate'] = 'GLM'
        avg_acc.loc[avg_acc['Model'] == 'SimpleLogistic_weka', 'Clf_cate'] = 'GLM'
        avg_acc.loc[avg_acc['Model'] == 'IBk_1_weka', 'Clf_cate'] = 'KNN'
        avg_acc.loc[avg_acc['Model'] == 'IBk_5_weka', 'Clf_cate'] = 'KNN'
        avg_acc.loc[avg_acc['Model'] == 'DecisionStump_weka', 'Clf_cate'] = 'DT'
        avg_acc.loc[avg_acc['Model'] == 'HoeffdingTree_weka', 'Clf_cate'] = 'DT'
        avg_acc.loc[avg_acc['Model'] == 'REPTree_weka', 'Clf_cate'] = 'DT'
        avg_acc.loc[avg_acc['Model'] == 'J48_weka', 'Clf_cate'] = 'DT'
        avg_acc.loc[avg_acc['Model'] == 'DecisionTable_weka', 'Clf_cate'] = 'DT'
        avg_acc.loc[avg_acc['Model'] == 'AdaBoostM1_weka', 'Clf_cate'] = 'Boosting'
        avg_acc.loc[avg_acc['Model'] == 'AttributeSelected_weka', 'Clf_cate'] = 'DT'


    ###check
    if len(avg_acc.loc[avg_acc['Clf_cate'] == -1])>0:
        print(Estimator_name, 'not classified')
        exit()

    Results_all = pd.concat([Results_all,avg_acc])

    return Results_all

def generate_data_set():
    Results_all = pd.DataFrame()
    sample_size_list = ['1k', '10k', '100k']
    Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']

    DATASET_list = ['NHTS', 'London', 'SG']

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
                           'KNN_python','MixL_biogeme','MixL_biogeme_RandSpec','MNL_biogeme','MNL_biogeme_RandSpec','NL_biogeme','NL_biogeme_RandSpec','SVM_python','BY_R','DA_R','DecisionTree_R',
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
                        output_file_path = 'Results/Results_London_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                        if os.path.exists(output_file_path):
                            Results = pd.read_csv(output_file_path)
                            Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                        else:
                            print(output_file_path, 'does not exist')
                    elif DATASET == 'SG':
                        output_file_path = 'Results/Results_SG_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                        if os.path.exists(output_file_path):
                            Results = pd.read_csv(output_file_path)
                            Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                        else:
                            print(output_file_path, 'does not exist')
                    else:
                        output_file_path = 'Results/Results_' + output_name + '_' + Estimator_name + '_' + sample_size + '.csv'
                        if os.path.exists(output_file_path):
                            Results = pd.read_csv(output_file_path)
                            Results_all = process_results(Dependent_var, DATASET, Estimator_name,sample_size, Results, Results_all)
                        else:
                            print(output_file_path, 'does not exist')
    # Results_all.loc[Results_all['Model'] == 'NL_biogeme','Run_time_5CV_second'] *= 60 # from minutes to seconds
    # Results_all.loc[Results_all['Model'] == 'MixL_biogeme', 'Run_time_5CV_second'] *= 60  # from minutes to seconds
    print('finish generating...')
    Results_all.to_csv('output/All_results.csv',index=False)




def classifer_big_cate(indicator, Results_all,save_fig):
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    # if indicator == 'Run_time_5CV_second':
    #     Results_all = Results_all.loc[Results_all['Sample_size']=='100k']
    Results_all['avg_acc'] = Results_all.groupby(['Clf_cate'])[indicator].transform('mean')
    Results_all['median_acc'] = Results_all.groupby(['Clf_cate'])[indicator].transform('median')
    Results_all['max_acc'] = Results_all.groupby(['Clf_cate'])[indicator].transform('max')
    Results_all = Results_all.sort_values(['avg_acc'],ascending=False)
    order_plot = list(pd.unique(Results_all['Clf_cate']))
    Results_all_sinlge = Results_all.loc[:,['Clf_cate','avg_acc','median_acc','max_acc']].drop_duplicates()
    ############
    font_size = 16
    plt.figure(figsize=(12, 8))
    X = list(range(len(Results_all_sinlge)))
    if indicator!= 'Accuracy':
        sns.violinplot(x = Results_all['Clf_cate'], y = Results_all[indicator], color="gray", cut=0)
    else:
        sns.violinplot(x=Results_all['Clf_cate'], y=Results_all[indicator], color="gray")
    sns.stripplot(x = Results_all['Clf_cate'], y = Results_all[indicator],color = 'white',size = 3)
    plt.scatter(x = X, y = Results_all_sinlge['avg_acc'],color = 'blue', label ='Mean',zorder = 5)
    plt.scatter(x=X, y=Results_all_sinlge['median_acc'], color='red', label ='Median',zorder = 4)
    if indicator == 'Accuracy':
        plt.scatter(x=X, y=Results_all_sinlge['max_acc'], color='tab:green', label ='Max',zorder = 4)

    plt.xlabel('Model families',fontsize=font_size)
    if indicator == 'Accuracy':
        plt.ylabel('Prediction accuracy',fontsize=font_size)
    else:
        plt.ylabel('Training + testing time (log scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(X, order_plot, fontsize=font_size, rotation = 90)
    x_lim = [X[0]-0.5, X[-1]+0.5]
    if indicator!= 'Accuracy':
        time_refer = [1, 3, 10, 60, 180, 600, 25*60, 3600, 4*3600]
        time_refer_log = np.log(time_refer)
        label_time = ['<1 sec', '3 sec','10 sec', '1 min', '3 min','10 min', '25 min','1 h','4 h']
        # for time, label_t in zip(time_refer, label_time):
        #     plt.plot([x_lim[0],x_lim[1]], [np.log(time), np.log(time)], 'k--')
        #     plt.text(X[-1]-0.5, np.log(time)+0.2, label_t, fontsize = font_size)
        plt.yticks(time_refer_log, label_time,fontsize=font_size)

        mean_text = list(np.exp(Results_all_sinlge['avg_acc']))
        mean_text = np.round(mean_text, 2)
        mean_text = [str(num) if num>1 else '<' + str(num) for num in mean_text ]
        median_text = list(np.exp(Results_all_sinlge['median_acc']))
        median_text = np.round(median_text, 2)
        median_text = [str(num) if num>1 else '<' + str(num) for num in median_text]
        max_text = list(np.exp(Results_all_sinlge['max_acc']))
        max_text = np.round(max_text, 1)
        max_text = [str(num) if num>1 else '<' + str(num) for num in max_text]

        count = 0
        for x_,tx in zip(X, mean_text):
            if count > 3:
                plt.text(x_-0.28, np.log(1.3*3600/5), tx, color='blue', fontsize=font_size)
            elif count == 0:
                plt.text(x_ - 0.3, np.log(6/2), tx, color='blue', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, np.log(6 * 3600/5), tx, color='blue', fontsize=font_size)
            count+=1
        count = 0
        for x_, tx in zip(X, median_text):
            if count >3:
                plt.text(x_-0.28, np.log(0.9*3600/5), tx, color='red', fontsize=font_size)
            elif count == 0:
                plt.text(x_ - 0.3, np.log(4/2), tx, color='red', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, np.log(3.9 * 3600/5), tx, color='red', fontsize=font_size)
            count+=1


        # count = 0
        # for x_, tx in zip(X, max_text):
        #     if count >3:
        #         plt.text(x_-0.28, np.log(0.6*3600/5), tx, color='tab:green', fontsize=font_size)
        #     elif count == 0:
        #         plt.text(x_ - 0.3, np.log(2.5/2), tx, color='tab:green', fontsize=font_size)
        #     else:
        #         plt.text(x_ - 0.28, np.log(2.5 * 3600/5), tx, color='tab:green', fontsize=font_size)
        #     count+=1

    plt.xlim(x_lim[0],x_lim[1])
    if indicator == 'Accuracy':
        y_lim = [0,1]
        plt.ylim(y_lim[0],y_lim[1])

    if indicator == 'Accuracy':
        mean_text = list(Results_all_sinlge['avg_acc'] * 100)
        mean_text = np.round(mean_text, 2)
        mean_text = [str(num) for num in mean_text]
        median_text = list(Results_all_sinlge['median_acc'] * 100)
        median_text = np.round(median_text, 2)
        median_text = [str(num) for num in median_text]
        max_text = list(Results_all_sinlge['max_acc'] * 100)
        max_text = np.round(max_text, 2)
        max_text = [str(num) for num in max_text]


        count = 0
        for x_, tx in zip(X, mean_text):
            if count == 0:
                plt.text(x_ - 0.28, 0.11, tx, color='blue', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.11, tx, color='blue', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.11, tx, color='blue', fontsize=font_size)
            count += 1
        count = 0
        for x_, tx in zip(X, median_text):
            if count == 0:
                plt.text(x_ - 0.28, 0.06, tx, color='red', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.06, tx, color='red', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.06, tx, color='red', fontsize=font_size)
            count += 1

        count = 0
        for x_, tx in zip(X, max_text):
            if count == 0:
                plt.text(x_ - 0.28, 0.01, tx, color='tab:green', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.01, tx, color='tab:green', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.01, tx, color='tab:green', fontsize=font_size)
            count += 1


    plt.legend(fontsize=font_size, ncol = 2)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/model_family_'+ indicator + '.png',dpi = 200)


def classife_small_cate(indicator, Results_all, save_fig):

    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    Results_all['avg_acc'] = Results_all.groupby(['Model'])[indicator].transform('mean')
    Results_all['median_acc'] = Results_all.groupby(['Model'])[indicator].transform('median')
    Results_all['acc_std'] = Results_all.groupby(['Model'])[indicator].transform('std')
    Results_all = Results_all.sort_values(['avg_acc'], ascending=False)

    Results_all_single = Results_all.loc[:,['Model','avg_acc','median_acc','acc_std']].drop_duplicates()
    X = list(range(len(Results_all_single)))
    #
    # Results_all_single.sort_values(['acc_std'])
    # ############
    font_size = 10
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.errorbar(X, Results_all_single['avg_acc'], yerr=Results_all_single['acc_std'], color = 'blue', fmt='o',label ='Mean and Std.')
    # plt.scatter(Results_all_single['Model'], Results_all_single['avg_acc'],color = 'red', label ='Mean',zorder = 5)
    plt.scatter(X, Results_all_single['median_acc'], color='red', label ='Median',zorder = 4)

    plt.xlabel('Classifiers', fontsize=font_size)

    if indicator == 'Accuracy':
        plt.ylabel('Prediction accuracy',fontsize=font_size)

    else:
        plt.ylabel('Training + testing time (log scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(X, Results_all_single['Model'], fontsize=font_size, rotation=90)
    if indicator == 'Accuracy':
        # ax = plt.gca()
        dcm_list = ['mnl_B','nl_B','mxl_B']
        Results_all_single = Results_all_single.reset_index(drop=True)
        for dcm_m in dcm_list:
            DCM_id = Results_all_single.loc[Results_all_single['Model']==dcm_m].index.to_numpy()[0]
            print(dcm_m, Results_all_single.loc[Results_all_single['Model'] == dcm_m])
            ax.get_xticklabels()[int(DCM_id)].set_color("red")
    else:
        dcm_list = ['mnl_B','nl_B','mxl_B','LogitBoost_R','Gradient Boosting_P']
        Results_all_single = Results_all_single.reset_index(drop=True)
        for dcm_m in dcm_list:
            DCM_id = Results_all_single.loc[Results_all_single['Model']==dcm_m].index.to_numpy()[0]
            print(dcm_m, Results_all_single.loc[Results_all_single['Model']==dcm_m])
            ax.get_xticklabels()[int(DCM_id)].set_color("red")

    if indicator!= 'Accuracy':
        time_refer = [1, 3, 10, 60, 180, 600, 25*60, 3600, 4*3600]
        time_refer_log = np.log(time_refer)
        label_time = ['<1 sec', '3 sec','10 sec', '1 min', '3 min','10 min', '25 min','1 h','4 h']
        # for time, label_t in zip(time_refer, label_time):
        #     plt.plot([x_lim[0],x_lim[1]], [np.log(time), np.log(time)], 'k--')
        #     plt.text(X[-1]-0.5, np.log(time)+0.2, label_t, fontsize = font_size)
        plt.yticks(time_refer_log, label_time,fontsize=font_size)

    x_lim = [X[0]-1, X[-1]+1]
    plt.xlim(x_lim[0],x_lim[1])
    # y_lim = [0, 1]
    # plt.ylim(y_lim[0], y_lim[1])
    plt.legend(fontsize=font_size)
    # if indicator!= 'Accuracy':
    #     time_refer = [60, 600, 3600, 4*3600]
    #     label_time = ['1 min','10 min','1 h','4 h']
    #     for time, label_t in zip(time_refer, label_time):
    #         plt.plot([x_lim[0],x_lim[1]], [np.log(time), np.log(time)], 'k--')
    #         plt.text(X[-1]-5, np.log(time)+0.2, label_t, fontsize = font_size)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/classifiers_' + indicator +'.png', dpi=200)
    a=1



def sample_size(indicator, Results_all, save_fig):
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    Results_all['avg_acc'] = Results_all.groupby(['Sample_size'])[indicator].transform('mean')
    Results_all['median_acc'] = Results_all.groupby(['Sample_size'])[indicator].transform('median')
    Results_all['max_acc'] = Results_all.groupby(['Sample_size'])[indicator].transform('max')
    Results_all = Results_all.sort_values(['avg_acc'], ascending=True)

    order_plot = list(pd.unique(Results_all['Sample_size']))
    Results_all_sinlge = Results_all.loc[:,['Sample_size','avg_acc','median_acc','max_acc']].drop_duplicates()
    X = list(range(len(Results_all_sinlge)))

    ############
    font_size = 16
    plt.figure(figsize=(8, 8))
    if indicator!= 'Accuracy':
        sns.violinplot(x = Results_all['Clf_cate'], y = Results_all[indicator], color="gray", cut=0)
    else:
        sns.violinplot(x = Results_all['Sample_size'], y = Results_all[indicator], color="gray",order = order_plot)
    sns.stripplot(x = Results_all['Sample_size'], y = Results_all[indicator],color = 'white',size = 3)
    plt.scatter(x = X, y = Results_all_sinlge['avg_acc'],color = 'blue', label ='Mean',zorder = 5)
    plt.scatter(x=X, y = Results_all_sinlge['median_acc'], color='red', label ='Median',zorder = 4)
    if indicator == 'Accuracy':
        plt.scatter(x=X, y = Results_all_sinlge['max_acc'], color='tab:green', label='Max', zorder=4)
    plt.xlabel('Sample size', fontsize=font_size)
    if indicator == 'Accuracy':
        print('sample size avg acc', Results_all_sinlge['avg_acc'])
        plt.ylabel('Prediction accuracy',fontsize=font_size)
    else:
        print('sample size avg running time', np.exp(Results_all_sinlge['avg_acc']))
        plt.ylabel('Training + testing time (log scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size, rotation=0)
    plt.legend(fontsize=font_size, ncol = 2)
    x_lim = [X[0]-0.5, X[-1]+0.5]
    plt.xlim(x_lim[0],x_lim[1])
    # if indicator!= 'Accuracy':
    #     time_refer = [60, 600, 3600, 4*3600]
    #     label_time = ['1 min','10 min','1 h','4 h']
    #     for time, label_t in zip(time_refer, label_time):
    #         plt.plot([x_lim[0],x_lim[1]], [np.log(time), np.log(time)], 'k--')
    #         plt.text(X[-1]-0.5, np.log(time)+0.2, label_t, fontsize = font_size)

    if indicator!= 'Accuracy':
        time_refer = [1, 3, 10, 60, 180, 600, 25*60, 3600, 4*3600]
        time_refer_log = np.log(time_refer)
        label_time = ['<1 sec', '3 sec','10 sec', '1 min', '3 min','10 min', '25 min','1 h','4 h']
        # for time, label_t in zip(time_refer, label_time):
        #     plt.plot([x_lim[0],x_lim[1]], [np.log(time), np.log(time)], 'k--')
        #     plt.text(X[-1]-0.5, np.log(time)+0.2, label_t, fontsize = font_size)
        plt.yticks(time_refer_log, label_time,fontsize=font_size)

        mean_text = list(np.exp(Results_all_sinlge['avg_acc']))
        mean_text = np.round(mean_text, 2)
        mean_text = [str(num) +' sec'for num in mean_text]
        median_text = list(np.exp(Results_all_sinlge['median_acc']))
        median_text = np.round(median_text, 2)
        median_text = [str(num) +' sec'for num in median_text]
        max_text = list(np.exp(Results_all_sinlge['max_acc']))
        max_text = np.round(max_text, 2)
        max_text = [str(num) +' sec'for num in max_text]

        count = 0
        for x_,tx in zip(X, mean_text):
            if count == 0:
                plt.text(x_-0.21, np.log(0.5*3600/5), tx, color='blue', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.21, np.log(1 * 3600/5), tx, color='blue', fontsize=font_size)
            else:
                plt.text(x_ - 0.21, np.log(3 * 3600/5), tx, color='blue', fontsize=font_size)
            count+=1

        count = 0
        for x_, tx in zip(X, median_text):
            if count == 0:
                plt.text(x_-0.21, np.log(0.3*3600/5), tx, color='red', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.21, np.log(0.65 * 3600/5), tx, color='red', fontsize=font_size)
            else:
                plt.text(x_ - 0.21, np.log(2 * 3600/5), tx, color='red', fontsize=font_size)
            count+=1


        # count = 0
        # for x_, tx in zip(X, max_text):
        #     if count == 0:
        #         plt.text(x_-0.21, np.log(0.3*3600/5), tx, color='tab:green', fontsize=font_size)
        #     elif count == 1:
        #         plt.text(x_ - 0.21, np.log(0.65 * 3600/5), tx, color='tab:green', fontsize=font_size)
        #     else:
        #         plt.text(x_ - 0.21, np.log(2 * 3600/5), tx, color='tab:green', fontsize=font_size)
        #     count+=1


        # elif 'SG' in idx[0]:
        #     for x_,tx in zip(X, mean_text):
        #         plt.text(x_-0.19, 0.1,tx, color='blue', fontsize=font_size)
        #     for x_, tx in zip(X, median_text):
        #         plt.text(x_ - 0.19, 0.03, tx, color='red', fontsize=font_size)
        # else:
        #     for x_,tx in zip(X, mean_text):
        #         plt.text(x_-0.28, 0.1,tx, color='blue', fontsize=font_size)
        #     for x_, tx in zip(X, median_text):
        #         plt.text(x_ - 0.28, 0.03, tx, color='red', fontsize=font_size)


    if indicator == 'Accuracy':
        mean_text = list(Results_all_sinlge['avg_acc']*100)
        mean_text = np.round(mean_text, 2)
        mean_text = [str(num) for num in mean_text]
        median_text = list(Results_all_sinlge['median_acc']*100)
        median_text = np.round(median_text, 2)
        median_text = [str(num) for num in median_text]
        max_text = list(Results_all_sinlge['max_acc'] * 100)
        max_text = np.round(max_text, 2)
        max_text = [str(num) for num in max_text]
        count = 0
        for x_,tx in zip(X, mean_text):
            if count == 0:
                plt.text(x_-0.12, 0.13, tx, color='blue', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.12,  0.13, tx, color='blue', fontsize=font_size)
            else:
                plt.text(x_ - 0.12,  0.13, tx, color='blue', fontsize=font_size)
            count+=1
        count = 0
        for x_, tx in zip(X, median_text):
            if count == 0:
                plt.text(x_-0.12, 0.08, tx, color='red', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.12, 0.08, tx, color='red', fontsize=font_size)
            else:
                plt.text(x_ - 0.12, 0.08, tx, color='red', fontsize=font_size)
            count+=1

        count = 0
        for x_, tx in zip(X, max_text):
            if count == 0:
                plt.text(x_ - 0.12, 0.03, tx, color='tab:green', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.12, 0.03, tx, color='tab:green', fontsize=font_size)
            else:
                plt.text(x_ - 0.12, 0.03, tx, color='tab:green', fontsize=font_size)
            count += 1


        y_lim = [0, 1]
        plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Sample_size_' + indicator + '.png', dpi=200)


def data_set(Results_all, save_fig):
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    Results_all = Results_all.loc[Results_all['Dependent_var'] == 'MODE']
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    Results_all['avg_acc'] = Results_all.groupby(['Data_set'])['Accuracy'].transform('mean')
    Results_all['median_acc'] = Results_all.groupby(['Data_set'])['Accuracy'].transform('median')
    Results_all = Results_all.sort_values(['avg_acc'], ascending=True)

    order_plot = list(pd.unique(Results_all['Data_set']))
    Results_all_sinlge = Results_all.loc[:,['Data_set','avg_acc','median_acc']].drop_duplicates()
    ############
    font_size = 16
    plt.figure(figsize=(8, 8))
    sns.violinplot(x = Results_all['Data_set'], y = Results_all['Accuracy'], color="gray",order = order_plot)
    sns.stripplot(x = Results_all['Data_set'], y = Results_all['Accuracy'],color = 'white',size = 3)
    plt.scatter(x = order_plot, y = Results_all_sinlge['avg_acc'],color = 'blue', label ='Mean',zorder = 5)
    plt.scatter(x=order_plot, y=Results_all_sinlge['median_acc'], color='red', label ='Median',zorder = 4)
    plt.xlabel('Date set', fontsize=font_size)
    plt.ylabel('Prediction accuracy', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size, rotation=0)
    plt.legend(fontsize=font_size)
    x_lim = [-0.5, 2.5]
    plt.xlim(x_lim[0],x_lim[1])
    y_lim = [0, 1]
    plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/data_set.png', dpi=200)



def depedent_var(Results_all, save_fig):
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    Results_all = Results_all.loc[Results_all['Data_set'] == 'NHTS']
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    Results_all['avg_acc'] = Results_all.groupby(['Dependent_var'])['Accuracy'].transform('mean')
    Results_all['median_acc'] = Results_all.groupby(['Dependent_var'])['Accuracy'].transform('median')
    Results_all = Results_all.sort_values(['avg_acc'], ascending=True)

    order_plot = list(pd.unique(Results_all['Dependent_var']))
    Results_all_sinlge = Results_all.loc[:,['Dependent_var','avg_acc','median_acc']].drop_duplicates()
    ############
    font_size = 16
    plt.figure(figsize=(8, 8))
    sns.violinplot(x = Results_all['Dependent_var'], y = Results_all['Accuracy'], color="gray",order = order_plot)
    sns.stripplot(x = Results_all['Dependent_var'], y = Results_all['Accuracy'],color = 'white',size = 3)
    plt.scatter(x = order_plot, y = Results_all_sinlge['avg_acc'],color = 'blue', label ='Mean',zorder = 5)
    plt.scatter(x=order_plot, y=Results_all_sinlge['median_acc'], color='red', label ='Median',zorder = 4)
    plt.xlabel('Dependent variables', fontsize=font_size)
    plt.ylabel('Prediction accuracy', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(order_plot, ['Travel mode','Trip purpose','Car ownership'], fontsize=font_size, rotation=0)
    plt.legend(fontsize=font_size)
    x_lim = [-0.5, 2.5]
    plt.xlim(x_lim[0],x_lim[1])
    y_lim = [0, 1]
    plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/dependent_var.png', dpi=200)


def data_set_fixed_dimension(Results_all, save_fig):
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    # 10k
    indicator = 'Accuracy'
    Results_all = Results_all.loc[Results_all['Dependent_var'] == 'MODE']
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '10k']
    data_set_list = ['SG','NHTS','London']
    font_size = 10
    plt.figure(figsize=(15, 8))
    count = -1
    Label_list = ['SG-MC','NHTS-MC','London-MC']

    for data_set in data_set_list:
        count += 1
        Results_used = Results_all.loc[Results_all['Data_set'] == data_set]
        Results_used['avg_acc'] = Results_used.groupby(['Model'])[indicator].transform('mean')
        Results_used['median_acc'] = Results_used.groupby(['Model'])[indicator].transform('median')
        Results_used['acc_std'] = Results_used.groupby(['Model'])[indicator].transform('std')
        Results_used = Results_used.sort_values(['avg_acc'], ascending=False)

        Results_all_single = Results_used.loc[:, ['Model', 'avg_acc', 'median_acc', 'acc_std']].drop_duplicates()
        if count == 0:
            X = list(range(len(Results_all_single)))
            Results_all_single['Model_id'] = X
            model_id = Results_all_single.loc[:,['Model','Model_id']]
        else:
            Results_all_single = Results_all_single.merge(model_id, on =['Model'])

            a=1
        #
        # ############
        plt.errorbar(Results_all_single['Model_id'], Results_all_single['avg_acc'], yerr=Results_all_single['acc_std'], color= colors[count], fmt='o',
                     label=Label_list[count], alpha = 1)
        # plt.scatter(Results_all_single['Model'], Results_all_single['avg_acc'],color = 'red', label ='Mean',zorder = 5)
        # plt.scatter(X, Results_all_single['median_acc'], color='red', label='Median', zorder=4)

    plt.xlabel('Classifiers', fontsize=font_size)

    if indicator == 'Accuracy':
        plt.ylabel('Prediction accuracy', fontsize=font_size)
    else:
        plt.ylabel('Training + testing time (log scale))', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(Results_all_single['Model_id'], Results_all_single['Model'], fontsize=font_size, rotation=90)
    x_lim = [X[0] - 1, X[-1] + 1]
    plt.xlim(x_lim[0], x_lim[1])
    # y_lim = [0, 1]
    # plt.ylim(y_lim[0], y_lim[1])
    plt.legend(fontsize=font_size)
    if indicator != 'Accuracy':
        time_refer = [60, 600, 3600, 4 * 3600]
        label_time = ['1 min', '10 min', '1 h', '4 h']
        for time, label_t in zip(time_refer, label_time):
            plt.plot([x_lim[0], x_lim[1]], [np.log(time), np.log(time)], 'k--')
            plt.text(X[-1] - 5, np.log(time) + 0.2, label_t, fontsize=font_size)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/data_set_' + indicator + '.png', dpi=200)
    a = 1

    #

    a=1

def classifier_in_top_prob(Results_all, save_fig):
    Results_all_used = Results_all.copy()
    Results_all_used = Results_all_used.sort_values(['Accuracy'],ascending=False)
    Results_all_group_task = Results_all_used.groupby(['Sample_size','Dependent_var','Data_set'])
    all_classifer = Results_all.loc[:,['Model']]
    all_classifer = all_classifer.drop_duplicates().reset_index(drop=True)

    all_classifer['top_10_count'] = 0
    all_classifer['top_5_count'] = 0
    all_classifer['top_20_count'] = 0
    task_num = 0
    for idx,info in Results_all_group_task:
        task_num += 1
        top_10_c = list(info.head(10)['Model'])
        top_5_c = list(info.head(5)['Model'])
        top_20_c = list(info.head(20)['Model'])
        all_classifer.loc[all_classifer['Model'].isin(top_10_c),'top_10_count'] += 1
        all_classifer.loc[all_classifer['Model'].isin(top_5_c), 'top_5_count'] += 1
        all_classifer.loc[all_classifer['Model'].isin(top_20_c), 'top_20_count'] += 1
    dcm_list = ['mnl_B', 'nl_B', 'mxl_B']
    all_classifer_used = all_classifer.loc[(~((all_classifer['top_20_count'] == 0)&
                                             (all_classifer['top_5_count'] == 0)&
                                             (all_classifer['top_10_count'] == 0))) | (all_classifer['Model'].isin(dcm_list))]



    all_classifer_used['top_20_prob'] = all_classifer_used['top_20_count'] / task_num
    all_classifer_used['top_10_prob'] = all_classifer_used['top_10_count'] / task_num
    all_classifer_used['top_5_prob'] = all_classifer_used['top_5_count'] / task_num
    all_classifer_used = all_classifer_used.sort_values(['top_20_prob'],ascending=False)
    all_classifer_used['Model_id'] = list(range(len(all_classifer_used)))

    font_size = 12
    fig, ax = plt.subplots(figsize=(15, 8))
    X = list(all_classifer_used['Model_id'])
    plt.plot(all_classifer_used['Model_id'],all_classifer_used['top_20_prob'],color=colors[0], linewidth=1.5, label='Top 20', marker = '^', markersize = 8)
    plt.plot(all_classifer_used['Model_id'], all_classifer_used['top_10_prob'], color=colors[1], linewidth=1.5,
             label='Top 10', marker='s', markersize=8)
    plt.plot(all_classifer_used['Model_id'], all_classifer_used['top_5_prob'], color=colors[2], linewidth=1.5,
             label='Top 5', marker='.', markersize=8)



    plt.xlabel('Classifiers', fontsize=font_size)
    plt.ylabel('Proportion of being top N classifiers', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(all_classifer_used['Model_id'], all_classifer_used['Model'], fontsize=font_size, rotation=90)


    dcm_list = ['mnl_B', 'nl_B', 'mxl_B']
    for dcm_m in dcm_list:
        if dcm_m not in list(all_classifer_used['Model']):
            continue
        DCM_id = all_classifer_used.loc[all_classifer_used['Model'] == dcm_m,'Model_id'].iloc[0]
        # print(len(ax.get_xticklabels()))
        ax.get_xticklabels()[int(DCM_id)].set_color("red")


    x_lim = [X[0] - 1, X[-1] + 1]
    plt.xlim(x_lim[0], x_lim[1])
    # y_lim = [0, 1]
    # plt.ylim(y_lim[0], y_lim[1])
    plt.legend(fontsize=font_size)

    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Proportion_of_being_top_N_classifiers' + '.png', dpi=200)

def every_data_set_top_10_classifier(Results_all, save_fig=1):
    data_set_list = ['NHTS','London','SG']
    Results_all_used = Results_all.loc[Results_all['Dependent_var']=='MODE']
    data_save = pd.DataFrame()
    for ds in data_set_list:
        Results_all_data_set = Results_all_used.loc[Results_all_used['Data_set']==ds]
        Results_all_data_set = Results_all_data_set.groupby(['Model']).agg({'Accuracy':['mean','std']})
        Results_all_data_set = Results_all_data_set.reset_index()
        Results_all_data_set.columns = Results_all_data_set.columns.droplevel()
        Results_all_data_set.columns = ['Model','Acc_mean','Acc_std']
        Results_all_data_set = Results_all_data_set.sort_values(['Acc_mean'],ascending=False)
        Results_all_data_set['rank'] = list(range(len(Results_all_data_set)))
        Results_all_data_set['rank'] += 1
        DCM = Results_all_data_set.loc[Results_all_data_set['Model'].str.contains("_B")]
        Results_all_data_set = Results_all_data_set.head(10)
        Results_all_data_set['Data_set'] = ds
        DCM['Data_set'] = ds
        Results_all_data_set = Results_all_data_set.append(DCM, sort=False)
        data_save = data_save.append(Results_all_data_set)
        a=1
    data_save.to_csv('output/every_data_set_top_10_classifier.csv',index=False)


def data_set_distribution(Results_all,save_fig=1):
    data_set_list = ['NHTS', 'London', 'SG']
    label_list = ['NHTS2017-MC', 'LTDS2015-MC', 'SGP2017-MC']
    Results_all_used = Results_all.loc[(Results_all['Dependent_var']=='MODE')] #&(Results_all['Sample_size']=='100k')
    font_size = 16
    color_id = 0
    sns.set(font_scale=1)
    sns.set_style("white", {"legend.frameon": True})
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for ds in data_set_list:
        Results_all_data_set = Results_all_used.loc[Results_all_used['Data_set']==ds]
        ##############

        sns.kdeplot(Results_all_data_set['Accuracy'], shade=True, color=colors[color_id],label = label_list[color_id],ax=ax1)

        # counts, bin_edges = np.histogram(Results_all_data_set['Accuracy'], bins=100,range = [0,1], normed=True)
        # cdf = np.cumsum(counts)
        # plot the cdf
        # plt.plot(bin_edges[1:], cdf / cdf[-1])
        # ax2.plot(bin_edges[1:], cdf / cdf[-1],color = colors[color_id], linestyle = 'dashed', linewidth=2)
        #
        # # ax2.hist(Results_all_data_set['Accuracy'], bins=100000, normed=1, histtype='step', cumulative=1,)
        mean = Results_all_data_set['Accuracy'].mean()
        median = Results_all_data_set['Accuracy'].median()
        # max_num = Results_all_data_set['Accuracy'].max()
        # plt.scatter(x=X, y=Results_all_sinlge['avg_acc'], color='blue', label='Mean', zorder=5)
        # plt.scatter(x=X, y=Results_all_sinlge['median_acc'], color='red', label='Median', zorder=4)
        ax1.axvline(mean, color = colors[color_id], linestyle='dashed', linewidth=2)
        ax1.axvline(median, color=colors[color_id], linestyle='dotted', linewidth=2)
        if color_id == 0:
            ax1.text(mean, 11.5, round(mean,3),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=font_size - 2, color=colors[color_id])
            ax1.text(median, 11.5, round(median,3),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=font_size - 2, color=colors[color_id])
        if color_id == 2:
            ax1.text(mean, 10.5, round(mean,3),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=font_size - 2, color=colors[color_id])
            ax1.text(median, 10.5, round(median,3),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=font_size - 2, color=colors[color_id])
        if color_id == 1:
            ax1.text(mean, 12, round(mean,3),
                     horizontalalignment='right', verticalalignment='center',
                     fontsize=font_size- 2, color=colors[color_id])
            ax1.text(median, 12, round(median,3),
                     horizontalalignment='left', verticalalignment='center',
                     fontsize=font_size- 2, color=colors[color_id])
        color_id += 1
    # plt.xlabel('Accuracy', fontsize=font_size)
    ax1.set_ylabel('Probability density', fontsize=font_size)
    ax1.set_xlabel('Accuracy', fontsize=font_size)
    # ax2.set_ylabel('Cumulative density', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    x_lim = [0.2,0.8]
    plt.xlim(x_lim[0], x_lim[1])
    # y_lim = [0, 1]
    # plt.ylim(y_lim[0], y_lim[1])
    # ax2.set_ylim([0,1.05])
    plt.legend(fontsize=font_size-1)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/KDE_.png', dpi=200)



def sample_size_by_different_tasks(Results_all, save_fig=1):
    indicator = 'Accuracy'
    # all_cate = list(pd.unique(Results_all['Clf_cate']))
    # Results_all = Results_all.loc[Results_all['Sample_size'] == '100k']
    Results_all_used = Results_all.copy()


    Results_all_used_group = Results_all_used.groupby(['Data_set','Dependent_var'])

    for idx, info in Results_all_used_group:
        ############
        font_size = 16

        info['avg_acc'] = info.groupby(['Sample_size'])[indicator].transform('mean')
        info['median_acc'] = info.groupby(['Sample_size'])[indicator].transform('median')
        info['max_acc'] = info.groupby(['Sample_size'])[indicator].transform('max')
        info = info.sort_values(['avg_acc'], ascending=True)
        order_plot = pd.unique(info['Sample_size'])
        X = list(range(len(order_plot)))
        Results_all_sinlge = info.loc[:, ['Sample_size', 'avg_acc', 'median_acc','max_acc']].drop_duplicates()
        plt.figure(figsize=(4, 4))
        sns.violinplot(x=info['Sample_size'], y=info[indicator], color="gray", order=order_plot)
        sns.stripplot(x=info['Sample_size'], y=info[indicator], color='white', size=3)
        plt.scatter(x=X, y=Results_all_sinlge['avg_acc'], color='blue', label='Mean', zorder=5)
        plt.scatter(x=X, y=Results_all_sinlge['median_acc'], color='red', label='Median', zorder=4)
        plt.scatter(x=X, y=Results_all_sinlge['max_acc'], color='tab:green', label='max', zorder=4)
        mean_text = list(Results_all_sinlge['avg_acc']*100)
        mean_text = np.round(mean_text, 2)
        mean_text = [str(num) for num in mean_text]
        median_text = list(Results_all_sinlge['median_acc']*100)
        median_text = np.round(median_text, 2)
        median_text = [str(num) for num in median_text]
        max_text = list(Results_all_sinlge['max_acc']*100)
        max_text = np.round(max_text, 2)
        max_text = [str(num) for num in max_text]


        if 'London' in idx[0]:
            for x_,tx in zip(X, mean_text):
                plt.text(x_-0.28, 0.93,tx, color='blue', fontsize=font_size)
            for x_, tx in zip(X, median_text):
                plt.text(x_ - 0.28, 0.86, tx, color='red', fontsize=font_size)
            for x_, tx in zip(X, max_text):
                plt.text(x_ - 0.28, 0.79, tx, color='tab:green', fontsize=font_size)

        elif 'SG' in idx[0]:
            for x_,tx in zip(X, mean_text):
                plt.text(x_-0.19, 0.17,tx, color='blue', fontsize=font_size)
            for x_, tx in zip(X, median_text):
                plt.text(x_ - 0.19, 0.1, tx, color='red', fontsize=font_size)
            for x_, tx in zip(X, max_text):
                plt.text(x_ - 0.19, 0.03, tx, color='tab:green', fontsize=font_size)
        else:
            for x_,tx in zip(X, mean_text):
                plt.text(x_-0.28, 0.17,tx, color='blue', fontsize=font_size)
            for x_, tx in zip(X, median_text):
                plt.text(x_ - 0.28, 0.1, tx, color='red', fontsize=font_size)
            for x_, tx in zip(X, max_text):
                plt.text(x_ - 0.28, 0.03, tx, color='tab:green', fontsize=font_size)

        # plt.text(X, [1] * 3,median_text, color='red', fontsize=font_size)
        plt.xlabel('Sample size', fontsize=font_size)
        if indicator == 'Accuracy':
            # print('sample size avg acc', info['avg_acc'])
            plt.ylabel('Prediction accuracy', fontsize=font_size)

        else:
            # print('sample size avg running time', np.exp(info['avg_acc']))
            plt.ylabel('Training + testing time (log scale))', fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(fontsize=font_size, rotation=0)

        if 'NHTS' in idx[0] and idx[1] == 'CAR_OWN':
            plt.legend(fontsize=font_size - 2, loc = 'upper left')
        else:
            plt.legend(fontsize=font_size - 2)

        x_lim = [X[0] - 0.5, X[-1] + 0.5]
        plt.xlim(x_lim[0], x_lim[1])
        if indicator != 'Accuracy':
            time_refer = [60, 600, 3600, 4 * 3600]
            label_time = ['1 min', '10 min', '1 h', '4 h']
            for time, label_t in zip(time_refer, label_time):
                plt.plot([x_lim[0], x_lim[1]], [np.log(time), np.log(time)], 'k--')
                plt.text(X[-1] - 0.5, np.log(time) + 0.2, label_t, fontsize=font_size)

        if indicator == 'Accuracy':
            y_lim = [0, 1]
            plt.ylim(y_lim[0], y_lim[1])
        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            plt.savefig('img/Sample_size_' + indicator + '_'+ idx[0] + '_'+ idx[1] + '.png', dpi=200)




def every_sample_size_top_10_classifier(Results_all, save_fig=1):
    data_set_list = ['1k','10k','100k']
    Results_all_used = Results_all.copy()
    data_save = pd.DataFrame()
    for ds in data_set_list:
        Results_all_data_set = Results_all_used.loc[Results_all_used['Sample_size']==ds]
        Results_all_data_set = Results_all_data_set.groupby(['Model']).agg({'Accuracy':['mean','std']})
        Results_all_data_set = Results_all_data_set.reset_index()
        Results_all_data_set.columns = Results_all_data_set.columns.droplevel()
        Results_all_data_set.columns = ['Model','Acc_mean','Acc_std']
        Results_all_data_set = Results_all_data_set.sort_values(['Acc_mean'],ascending=False)
        Results_all_data_set['rank'] = list(range(len(Results_all_data_set)))
        Results_all_data_set['rank'] += 1
        DCM = Results_all_data_set.loc[Results_all_data_set['Model'].isin(["mnl_B", "nl_B"])]
        DCM['Sample_size'] = ds

        DNN_intended = Results_all_data_set.loc[Results_all_data_set['Model'] == "DNN_5_200_P"]
        DNN_intended['Sample_size'] = ds

        Results_all_data_set = Results_all_data_set.head(10)
        Results_all_data_set['Sample_size'] = ds
        Results_all_data_set = Results_all_data_set.append(DCM, sort=False)

        if 'DNN_5_200_P' not in list(Results_all_data_set['Model']):
            Results_all_data_set = Results_all_data_set.append(DNN_intended, sort=False)

        Results_all_data_set = Results_all_data_set.sort_values(['rank'])
        data_save = data_save.append(Results_all_data_set)

        # a=1
    data_save.to_csv('output/every_sample_size_top_10_classifier.csv',index=False)



def every_outputs_top_10_classifier(Results_all, save_fig=1):
    data_set_list = ['MODE','TRIPPURP','CAR_OWN']
    # Results_all_used = Results_all.loc[Results_all['Data_set']=='NHTS']
    Results_all_used = Results_all.copy()
    data_save = pd.DataFrame()
    for ds in data_set_list:
        Results_all_data_set = Results_all_used.loc[Results_all_used['Dependent_var']==ds]
        Results_all_data_set = Results_all_data_set.groupby(['Model']).agg({'Accuracy':['mean','std']})
        Results_all_data_set = Results_all_data_set.reset_index()
        Results_all_data_set.columns = Results_all_data_set.columns.droplevel()
        Results_all_data_set.columns = ['Model','Acc_mean','Acc_std']
        Results_all_data_set = Results_all_data_set.sort_values(['Acc_mean'],ascending=False)
        Results_all_data_set['rank'] = list(range(len(Results_all_data_set)))
        Results_all_data_set['rank'] += 1
        DCM = Results_all_data_set.loc[Results_all_data_set['Model'].str.contains("_B")]
        DCM['Sample_size'] = ds
        Results_all_data_set = Results_all_data_set.head(10)
        Results_all_data_set['Sample_size'] = ds
        Results_all_data_set = Results_all_data_set.append(DCM, sort=False)
        data_save = data_save.append(Results_all_data_set)
        a=1
    data_save.to_csv('output/every_outputs_top_10_classifier.csv',index=False)

def change_model_name_to_paper(Results_all):
    name_list = pd.read_csv('output/model_name_list_code_out_manually_processed.csv')
    name_list = name_list.loc[:, ['Model','latex_clf_cate','Model_name_latex']]
    Results_all = Results_all.merge(name_list, on = ['Model'])
    Results_all['Model'] = Results_all['Model_name_latex']
    Results_all['Clf_cate'] = Results_all['latex_clf_cate']
    return Results_all

if __name__ == '__main__':
    ####################
    generate_data_set()
    ########################



    Results_all = pd.read_csv('output/All_results.csv')
    Results_all = change_model_name_to_paper(Results_all)

    Results_all['Run_time_log'] = np.log(Results_all['Run_time_5CV_second']/5) # 5-cross_validate
    Results_all.loc[Results_all['Run_time_log']<0,'Run_time_log'] = 0
    # indicator_list = ['Accuracy','Run_time_log']
    # indicator_list = ['Run_time_log']
    indicator_list = ['Run_time_log']
    #

    for indicator in indicator_list:
        used_results = Results_all.copy()
        # classifer_big_cate(indicator, used_results,save_fig = 1)
        # classife_small_cate(indicator, used_results, save_fig = 1)
        sample_size(indicator, used_results, save_fig = 1)
    # # used_results = Results_all.copy()
    # data_set(used_results, save_fig= 1)
    # used_results = Results_all.copy()
    # depedent_var(used_results, save_fig = 1)

    # data_set_fixed_dimension(Results_all, save_fig = 0)

    ########################################
    # classifier_in_top_prob(Results_all,save_fig=1)


    ########################################
    # every_data_set_top_10_classifier(Results_all,save_fig=1)
    # data_set_distribution(Results_all, save_fig=0)

    ################
    # every_sample_size_top_10_classifier(Results_all,save_fig=1)
    # sample_size_by_different_tasks(Results_all, save_fig=1)


    ################
    # every_outputs_top_10_classifier(Results_all, save_fig=1)