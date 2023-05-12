import pandas as pd
import weka.core.jvm as jvm
jvm.start()
from weka.classifiers import Classifier
from weka.core.dataset import create_instances_from_matrices, Attribute
import numpy as np
import time
import os
from weka.classifiers import Evaluation
from weka.core.classes import Random
from sklearn.model_selection import KFold
from weka.filters import Filter



class Classifier_weka:
    def __init__(self, raw_data, data_test, col_cost, num_points, MODE_WANT, Output_name, Dependent_var,Re_run, computer_info, n_jobs, models):
        self.raw_data = raw_data
        self.Output_name = Output_name
        self.Dependent_var = Dependent_var
        self.computer_info = computer_info
        self.n_jobs = n_jobs
        self.Re_run = Re_run
        self.model = models
        self.cv_time = 5

        self.data_test = data_test
        self.col_cost = col_cost
        self.num_points = num_points
        self.MODE_WANT = MODE_WANT

    def data_precocess(self):
        # pure numeric
        y_columns = [self.Dependent_var]
        x_columns = list(self.raw_data.columns)
        x_columns.remove(self.Dependent_var)
        value = self.raw_data[self.Dependent_var].value_counts()
        self.base = value.max()/len(self.raw_data)
        self.X = np.array(self.raw_data.loc[:,x_columns])
        self.Y = np.array(self.raw_data.loc[:,y_columns]).reshape(-1,)


        print('X shape', self.X.shape)
        print('Y shape',self.Y.shape)


        self.X_test = np.array(self.data_test.loc[:,x_columns])
        self.Y_test = np.array(self.data_test.loc[:,y_columns]).reshape(-1,)
        self.col_cost_id = x_columns.index(self.col_cost)



    def change_to_weka(self,X,Y):
        weka_dataset = create_instances_from_matrices(X, Y, name="generated from matrices")
        # weka_dataset.class_is_last()
        nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        nominal.inputformat(weka_dataset)
        weka_dataset = nominal.filter(weka_dataset)
        weka_dataset.class_is_last()
        return weka_dataset

    def change_to_weka_only_X(self,X):
        weka_dataset_X = create_instances_from_matrices(X, name="generated from matrices")
        return weka_dataset_X

    def excute(self):

        tic = time.time()
        self.data_precocess()
        print ('process data time:', round(time.time() - tic,1))
        # define model


        # intial output table

        output_file_path = self.Output_name

        # if self.Re_run:
        #     Results = pd.DataFrame(
        #         {'Model': [], 'Fold': [], 'Accuracy': [],  'base': [], 'Computer_info': [],
        #          'n_jobs': [], 'Run_time_5CV_second': [],})
        #     Results.to_csv(output_file_path, index=False)
        # else:
        #     if not os.path.exists(output_file_path):
        #         Results = pd.DataFrame(
        #             {'Model': [], 'Fold': [], 'Accuracy': [], 'base': [], 'Computer_info': [],
        #              'n_jobs': [], 'Run_time_5CV_second': [], })
        #         Results.to_csv(output_file_path, index=False)
        #     else:
        #         Results = pd.read_csv(output_file_path)
        # train models

        Results = pd.DataFrame()

        Model_dict = {**self.model}
        print (Model_dict)

        for name in Model_dict:
            if '100k' in output_file_path:
                if name in ['BayesNet_weka']:
                    print(name, 'does not fit for 100k, skip it')
                    continue
            try:
                # if not self.Re_run:
                    # model_in_results = Results.loc[Results['Model']==name]
                    # if len(model_in_results) >= self.cv_time+1: # cv+avg
                    #     print(name, 'exist, skip...')
                    #     continue
                model = Model_dict[name]
                cls = Classifier(classname=model[0], options=model[1])
                print("Training model ", name, " ...")
                # kf = KFold(n_splits=self.cv_time, random_state=1)
                # accuracy = []
                tic = time.time()

                X_input = self.X
                Y_input = self.Y
                X_test = self.X_test
                Y_test = self.Y_test

                weka_train_data = self.change_to_weka(X_input, Y_input)
                cls.build_classifier(weka_train_data)

                min_X = np.min(X_test[:, self.col_cost_id])
                max_X = np.max(X_test[:, self.col_cost_id])

                all_points = np.linspace(min_X, max_X, num = self.num_points)

                prop_list = []

                for point_value in all_points:
                    x_temp = X_test.copy()
                    x_temp[:,self.col_cost_id] = point_value

                    # predict
                    weka_test_data = self.change_to_weka(x_temp,Y_test)
                    y_pred = cls.distributions_for_instances(weka_test_data)
                    y_prob = y_pred[:, self.MODE_WANT - 1]
                    prop = np.mean(y_prob)

                    prop_list.append(prop)

                Results = Results.append(pd.DataFrame({'Model': [name] * self.num_points, 'x_values':all_points,'pred_prop':prop_list}))



                # for train_index, test_index in kf.split(self.X):
                #     # if count > 1:
                #     #     continue
                #     X_train = self.X[train_index, :]
                #     Y_train = self.Y[train_index]
                #     X_test = self.X[test_index, :]
                #     Y_test = self.Y[test_index]
                #     weka_train_data = self.change_to_weka(X_train,Y_train)
                #     cls.build_classifier(weka_train_data)
                #     # predict
                #     weka_test_data = self.change_to_weka(X_test,Y_test)
                #     evl = Evaluation(weka_test_data)
                #     _ = evl.test_model(cls, weka_test_data)
                #     acc = evl.percent_correct
                #     print('current acc', acc)
                #     accuracy.append(acc/100)
                # Training_time = round((time.time() - tic), 2)
                # for cv_num in range(len(accuracy)):
                #     Results = pd.concat([Results, pd.DataFrame({'Model':[name],'Fold':['Fold'+str(cv_num+1)],'Accuracy': [accuracy[cv_num]],'Computer_info':[self.computer_info],'n_jobs':[self.n_jobs],
                #                              'base':[self.base],'Run_time_5CV_second':[Training_time]})],sort=False)
                # # save in every iteration
                # avg_acc = np.mean(accuracy)
                # Results = pd.concat([Results, pd.DataFrame(
                #     {'Model': [name],'Fold': ['Average'],'Accuracy': [avg_acc], 'Computer_info': [self.computer_info], 'n_jobs': [self.n_jobs],
                #        'base': [self.base], 'Run_time_5CV_second': [Training_time]})], sort=False)

                Results.to_csv(output_file_path,index=False)

            except:
                print('Model', name, 'is not applicable')
                raise



if __name__ == '__main__':
    tic = time.time()
    # Parameters:
    #=====================model============
    models = {'BayesNet_weka': ['weka.classifiers.bayes.BayesNet',[]],
              'NaiveBayes_weka': ['weka.classifiers.bayes.NaiveBayes',[]],
              'MLP_weka': ['weka.classifiers.functions.MultilayerPerceptron', ['-N', '50', '-H', 'a']],
              'Logistic_weka': ['weka.classifiers.functions.Logistic', []],
              'SimpleLogistic_weka': ['weka.classifiers.functions.SimpleLogistic', []],
              'IBk_1_weka': ['weka.classifiers.lazy.IBk', ['-K', '1']],
              'IBk_5_weka': ['weka.classifiers.lazy.IBk', ['-K', '5']],
              'DecisionStump_weka': ['weka.classifiers.trees.DecisionStump', []],
              'HoeffdingTree_weka': ['weka.classifiers.trees.HoeffdingTree', []],
              'REPTree_weka': ['weka.classifiers.trees.REPTree', []],
              'J48_weka': ['weka.classifiers.trees.J48', []],
              'DecisionTable_weka': ['weka.classifiers.rules.DecisionTable', []],
              'AdaBoostM1_weka': ['weka.classifiers.meta.AdaBoostM1', []],
              'AttributeSelected_weka': ['weka.classifiers.meta.AttributeSelectedClassifier', []],
              } # name, option
    # 'OneR_weka': ['weka.classifiers.rules.OneR', ['-B', '6']], does not work
    #  'Bagging_REP_weka': ['weka.classifiers.meta.Bagging', []],
    #===================================
    Estimator_name = 'Weka_python'

    computer_info = 'I9-9900K'
    sample_size_list = ['10k']
    # sample_size_list = ['100k']

    Re_run = True  # True: rerun all models, False: if there are results existed, jump it
    n_jobs = 1
    Dependent_var_list = ['MODE']
    # Dependent_var_list = ['MODE']
    #
    DATASET_list = ['London']
    # DATASET_list = ['London']



    col_cost = 'dur_pt_access' # 'cost_driving_fuel' # cost_driving_ccharge dur_pt_access
    num_points = 100

    MODE_WANT = 3 #4 # drive  1: walk, 2: cycle, 3: public transport, 4:drive
    # (1: walk, 2: cycle, 3: public transport, 4:drive)



    for DATASET in DATASET_list:
        for Dependent_var in Dependent_var_list:
            if DATASET == 'London' or DATASET == 'SG':
                if Dependent_var != 'MODE':
                    continue
            if Dependent_var == 'MODE':
                output_name = 'MC'
                data_name = 'mode_choice'
            elif Dependent_var == 'CAR_OWN':
                output_name = 'CO'
                data_name = 'car_ownership'
            else:
                output_name = 'TP'
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
                print('Read raw data time:', round(time.time() - tic, 1), 's')
                print('Current scenario', output_file_path)

                output_file_path = output_file_path.replace('.csv', '_' + col_cost + '_' + 'mode' + str(
                    int(MODE_WANT)) + '.csv')


                m = Classifier_weka(data, data_test, col_cost, num_points, MODE_WANT, output_file_path, Dependent_var, Re_run, computer_info, n_jobs, models)
                m.excute()
    ###############END
    jvm.stop()
    ######################