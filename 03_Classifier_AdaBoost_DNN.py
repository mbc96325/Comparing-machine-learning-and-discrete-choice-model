###############
# to run this code, please use keras version >= 2.2.4
##################

import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
import time
import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
from tensorflow.keras import layers
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier



class setup:
    def __init__(self, DNN_para):


        self.numHlayers = DNN_para['numHlayers']
        self.numUnits = [DNN_para['numUnits']] * self.numHlayers
        self.dropout_rate = DNN_para['dropout_rate'] # 0
        self.wd = DNN_para['wd']
        self.learning_rate = DNN_para['learning_rate']
        self.training_epochs = DNN_para['training_epochs']


        
def validate(data, target, pred_func, cost_func, sess, x, y, train):
    prediction, loss = sess.run([pred_func, cost_func], feed_dict = {x:data, y:target, train:False}) 
    correct = 0
    for t, p in zip (target, prediction):
        if (t == p):
            correct = correct + 1
    
    return loss, correct / len(target)

def nn_setup(numClasses, inputDim, numHlayers, numUnits, dropout_rate, wd):


    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    for k in range(numHlayers):
        if k == 0:
            model.add(Dense(numUnits[k], activation='relu',input_shape=(inputDim,)))
        else:
            model.add(Dense(numUnits[k], activation='relu'))
    # Add an output layer with numClasses output units:
    model.add(Dense(numClasses,activation='softmax'))

    return model



def DNN_estimate(x_train_nn_all, choice_train_nn_all, cv, base, output_file_path,
                 DNN_para, Re_run, computer_info,verbose_or_not, training_epochs,
                 n_estimator,Estimator_name, n_jobs):


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


    numAlts = len(np.unique(choice_train_nn_all))
    current_setup = setup(DNN_para)

    def get_model():
        model = nn_setup(numAlts, np.size(x_train_nn_all,axis=1),
                         current_setup.numHlayers, current_setup.numUnits,
                         current_setup.dropout_rate, current_setup.wd)

        if current_setup.learning_rate == -1:
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

        else:
            sgd = SGD(lr=current_setup.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

        return model

    def evaluate(est_y,true_y):
        total = len(est_y)
        true = len(est_y[np.where(est_y-true_y==0)])
        return true/total
    np.random.seed(1)
    # sample seperate:
    count = 0
    kf = KFold(n_splits=cv)
    accuracy = []
    name = Estimator_name
    tic = time.time()
    for train_index, test_index in kf.split(x_train_nn_all):
        model = get_model()
        count +=1
        x_train_nn = x_train_nn_all[train_index]
        choice_train_nn = choice_train_nn_all[train_index]
        x_test_nn = x_train_nn_all[test_index]
        choice_test_nn = choice_train_nn_all[test_index]



        batch_size = np.min([int(len(x_train_nn) / 10), 3000])
        num_batches = int(len(x_train_nn) / batch_size)

        print('[process] Cross-validation', count, 'Setting up DNN...')

        # train_y = tf.keras.utils.to_categorical(choice_train_nn-1, num_classes=numAlts)
        # test_y = tf.keras.utils.to_categorical(choice_test_nn-1, num_classes=numAlts)


        def simple_model():
            return model
        ann_estimator = KerasClassifier(build_fn=simple_model) # epochs=training_epochs, batch_size=batch_size, verbose=verbose_or_not
        # ann_estimator.fit(x_train_nn, choice_train_nn)
        boosted_ann = AdaBoostClassifier(base_estimator=ann_estimator, n_estimators= n_estimator, baichuan=1) # define baichuan,
        # cancel the sample weight normalization in the sklearn code which distracts DNN training

        boosted_ann.fit(x_train_nn, choice_train_nn, sample_weight = np.ones(len(choice_train_nn)))  #
        y_predict_test = boosted_ann.predict(x_test_nn)
        y_predict_train = boosted_ann.predict(x_train_nn)
        acc_train = evaluate(y_predict_train, choice_train_nn)
        acc_test = evaluate(y_predict_test, choice_test_nn)

        # train_results = model.fit(x_train_nn, train_y)# epochs=training_epochs, batch_size=batch_size,verbose=verbose_or_not
        # results  = model.evaluate(x_test_nn, test_y, batch_size=batch_size)
        # results_train = model.evaluate(x_train_nn, train_y, batch_size=batch_size)

        print ('training_acc:', acc_train)
        correct_test = acc_test
        accuracy.append(acc_test)
        print('test_acc:', np.max(acc_test))
        print('Cross-validation:',count,', Elapsed time is %s seconds' % (time.time() - tic))
        print('base acc', base)
        # Plotting
        # if plot_or_not:
        #     fig, ax = plt.subplots(figsize=[14,5])
        #     ax.plot(loss_t, 'g-', label = 'training loss')
        #     ax.plot(loss_test, 'r-', label = 'test loss')
        #
        #     ax.set_ylabel('Cross Entropy Loss')
        #     ax.legend()
        #     ax.set_xlabel('epochs')
        #     ax.set_title(" Hidden Layers = " + str(current_setup.numHlayers) +\
        #                   " Hidden units = " + str(current_setup.numUnits) + " Dropout = " + str(current_setup.dropout_rate) +\
        #                   " Weight decay = " + '{:.4f}'.format(current_setup.wd))
        #     plt.show()


    Training_time = round((time.time() - tic), 2)
    for cv_num in range(len(accuracy)):
        Results = pd.concat([Results, pd.DataFrame(
            {'Model': [name], 'Fold': ['Fold' + str(cv_num + 1)], 'Accuracy': [accuracy[cv_num]],
             'Computer_info': [computer_info], 'n_jobs': [n_jobs],
             'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)
    # save in every iteration
    avg_acc = np.mean(accuracy)
    Results = pd.concat([Results, pd.DataFrame(
        {'Model': [name], 'Fold': ['Average'], 'Accuracy': [avg_acc], 'Computer_info': [computer_info],
         'n_jobs': [n_jobs],
         'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)

    # save in every iteration

    Results.to_csv(output_file_path,index=False)




if __name__ == '__main__':

    tic = time.time()
    # Parameters:


    computer_info = 'I9-9900K'
    sample_size_list = ['1k', '10k', '100k']

    Re_run = False # True: rerun all models, False: if there are results existed, jump it

    Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']
    # Dependent_var_list = ['MODE']
    verbose_or_not = False
    dnn_struc_list = [(1,30),(3,30),(5,30),
                      (1,100),(3,100),(5,100),
                      (1,200),(3,200),(5,200)]

    # dnn_struc_list = [(5,200)]

    DATASET_list = ['NHTS', 'London', 'SG']
    # DATASET_list = ['NHTS']
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
            for dnn_struc in dnn_struc_list:
                #DNN specific
                numHlayers = dnn_struc[0]
                numUnits = dnn_struc[1]
                # not useful

                Estimator_name = 'DNN_AdaBoost' + '_' + str(numUnits) + '_' + str(numHlayers) + '_python'

                dropout_rate = -1  # 0
                wd = -1  #
                ############
                learning_rate = -1 #0.00003 #-1 # 0.0005 # -1 means default learning rate

                training_epochs = 50
                n_estimator = 5

                DNN_para = {'numHlayers':numHlayers, 'numUnits':numUnits, 'wd': wd,
                            'learning_rate':learning_rate,
                            'dropout_rate':dropout_rate, 'training_epochs': training_epochs}

                for sample_size in sample_size_list:
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
                    print('Read raw data time:', round(time.time() - tic, 1),'s')
                    print(data.values.shape)
                    value = data[Dependent_var].value_counts()
                    base = value.max() / len(data)
                    print('Read raw data time:', round(time.time() - tic, 1), 's')
                    x_train_nn_all = np.array(data.iloc[:, 1:])

                    num_cross_validation = 5
                    choice_train_nn_all = np.array(data.iloc[:, 0])
                    DNN_estimate(x_train_nn_all, choice_train_nn_all, num_cross_validation, base, output_file_path, DNN_para, Re_run, computer_info,
                                 verbose_or_not,training_epochs,n_estimator,Estimator_name, n_jobs=1)
                    print('Elapsed time is %s seconds' % (time.time() - tic))




