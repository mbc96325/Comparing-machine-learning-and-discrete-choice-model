###############
# to run this code, please use keras version == 2.2.4
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
from typing import List, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

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


def create_nn(para, learning_rate) -> Sequential:
    numClasses, inputDim, numHlayers, numUnits, dropout_rate, wd = \
        para[0], para[1], para[2],para[3],para[4],para[5]
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    for k in range(numHlayers):
        if k == 0:
            model.add(Dense(numUnits[k], activation='relu',input_shape=(inputDim,)))
        else:
            model.add(Dense(numUnits[k], activation='relu'))
    # Add an output layer with numClasses output units:
    model.add(Dense(numClasses,activation='softmax'))

    if learning_rate == -1:
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    else:
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    return model


def one_boost_step(model, X, y, r, lr) -> Tuple[float]:
    """
    parameters:
        r: residual
        lr: learning rate
    """
    r2 = model.predict(X)  # model is trained to output residual
    y2 = y - r + lr * r2  # y2: new approximation of y
    r = y - y2
    return r, y2


def one_boost_step_test():
    import numpy as np
    class Model():
        def __init__(self, output):
            self.output = output  # the model always outputs 'output'

        def predict(self, X):
            return np.ones((X.shape[0], X.shape[-1])) * self.output

    X = np.array([1, 2])
    X = X.reshape(1, 2, 1)
    y = np.array([1]).reshape(1, 1)
    model = Model(.25 - .05)
    r = np.array([.25]).reshape(1, 1)
    lr = .1
    rr, yy2 = one_boost_step(model, X, y, r, lr)
    np.testing.assert_approx_equal(rr[0, 0], .23)
    np.testing.assert_approx_equal(yy2[0, 0], .77)


class Nngboost(BaseEstimator, RegressorMixin):
    def __init__(self, para, n_estimators, nn_lr,epochs, batch_size):
        """
        paramaters:
            lr: learning rate of the boosting method
        """

        self.para = para
        self.n_estimators = n_estimators
        self.lr = 0.1
        self.lr_decay_const = 70
        self.batch_size = batch_size
        self.epochs = epochs
        self.nn_lr = nn_lr


    def fit(self, X, y):
        patience = 5  # patience for early stopping
        callback = [EarlyStopping(patience=patience, verbose=1)]
        print(f'early stopping with patience {patience}')
        self.nn_: List[Sequential] = []
        self.graph_: List[tf.Graph] = []
        self.sess_: List[tf.Session] = []
        r2_list = []
        r = y
        for i in range(self.n_estimators):
            print('~~~~~~~~~~~~~~~~~~~')
            print(f'estimator number {i}:')
            print('learning rate is : ', self.lr_updated(i))
            self.graph_.append(tf.Graph())
            self.sess_.append(tf.Session(graph=self.graph_[-1]))
            # with self.graph_[-1].as_default():
            # with tf.Session(graph=self.graph_[-1]):
            with self.graph_[i].as_default():
                with self.sess_[i].as_default():
                    self.nn_.append(create_nn(self.para, self.nn_lr))
                    history = self.nn_[-1].fit(x=X, y=r, epochs=self.epochs, batch_size=self.batch_size, callbacks=callback,
                                               verbose=0)
                    r, y2 = one_boost_step(self.nn_[-1], X, y, r, self.lr_updated(i))
                    r2_train = r2_score(y, y2)
                    r2_list.append(r2_train)
            print('r2 train: ',r2_train)

        self.n_estimators_best_ = np.argmax(r2_list) + 1
        print('best number of estimators based on validation r2: ', self.n_estimators_best_)
        return self

    def lr_updated(self, itr):
        return self.lr * 2 ** (-itr / self.lr_decay_const)

    def predict(self, X, n_estimators=None):
        # check_is_fitted(self, ['nn_', 'n_estimators_best_', 'r2_val_list_'])
        # X = check_array(X)
        if not n_estimators:
            n_estimators = self.n_estimators_best_
        print(f'using {n_estimators} for prediction')
        # if len(X.shape) != 3:
        #     raise ValueError('the dimension of input must be 3 ')
        # with tf.Session(graph=self.graph_[0]):
        with self.graph_[0].as_default():
            with self.sess_[0].as_default():
                y = self.nn_[0].predict(X)
        for i in range(1, n_estimators):
            # with  tf.Session(graph=self.graph_[i]):
            with self.graph_[i].as_default():
                with self.sess_[i].as_default():
                    y += self.lr_updated(i) * self.nn_[i].predict(X)
        return y

def evaluate(est_y,true_y):
    total = len(est_y)
    true = len(est_y[np.where(est_y-true_y==0)])
    return true/total

def DNN_estimate(x_train_nn_all, data_test, col_cost, num_points, MODE_WANT,  choice_train_nn_all, cv, base, output_file_path,
                 DNN_para, Re_run, computer_info,verbose_or_not, training_epochs,
                 n_estimator, Estimator_name, n_jobs):

    x_columns = list(data_test.columns)
    col_cost_id = x_columns.index(col_cost)


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


    numAlts = len(np.unique(choice_train_nn_all))
    current_setup = setup(DNN_para)





    np.random.seed(1)
    # sample seperate:
    count = 0
    # kf = KFold(n_splits=cv)
    accuracy = []

    name = Estimator_name
    tic = time.time()

    #for train_index, test_index in kf.split(x_train_nn_all):

    # count +=1

    x_train_nn = x_train_nn_all.copy()
    choice_train_nn = choice_train_nn_all.copy()


    x_test_nn = np.array(data_test.iloc[:, 1:])
    choice_test_nn =  np.array(data_test.iloc[:, 0])


    batch_size = np.min([int(len(x_train_nn) / 10), 3000])
    num_batches = int(len(x_train_nn) / batch_size)


    nn_para = [numAlts, np.size(x_train_nn_all, axis=1),
                     current_setup.numHlayers, current_setup.numUnits,
                     current_setup.dropout_rate, current_setup.wd]
    NG_model = Nngboost(nn_para, n_estimators=n_estimator, nn_lr=current_setup.learning_rate,
                        epochs=training_epochs, batch_size=batch_size)

    print('[process] Cross-validation', count, 'Setting up DNN...')

    train_y = tf.keras.utils.to_categorical(choice_train_nn - 1, num_classes=numAlts)
    test_y = tf.keras.utils.to_categorical(choice_test_nn - 1, num_classes=numAlts)

    NG_model.fit(x_train_nn, train_y)

    min_X = np.min(x_test_nn[:, col_cost_id])
    max_X = np.max(x_test_nn[:, col_cost_id])

    all_points = np.linspace(min_X, max_X, num=num_points)

    Results = pd.DataFrame()

    prop_list = []

    for point_value in all_points:

        x_temp = x_test_nn.copy()
        x_temp[:, col_cost_id] = point_value

        y_pred_prob = NG_model.predict(x_temp)
        prop = np.mean(y_pred_prob[:,MODE_WANT-1])


        # y_pred = np.argmax(y_pred_prob,axis =1 ) + 1 # index start from zero, label start from 1
        # prop = sum(y_pred == MODE_WANT) / len(y_pred)
        prop_list.append(prop)




    #     y_predict_test = NG_model.predict(x_test_nn)
    #     y_predict_train = NG_model.predict(x_train_nn)
    #     y_predict_test = np.argmax(y_predict_test, axis=1) + 1
    #     y_predict_train = np.argmax(y_predict_train, axis=1) + 1
    #     acc_train = evaluate(y_predict_train, choice_train_nn)
    #     acc_test = evaluate(y_predict_test, choice_test_nn)
    #
    #     # train_results = model.fit(x_train_nn, train_y)# epochs=training_epochs, batch_size=batch_size,verbose=verbose_or_not
    #     # results  = model.evaluate(x_test_nn, test_y, batch_size=batch_size)
    #     # results_train = model.evaluate(x_train_nn, train_y, batch_size=batch_size)
    #
    #     print ('training_acc:', acc_train)
    #     correct_test = acc_test
    #     accuracy.append(acc_test)
    #     print('test_acc:', np.max(acc_test))
    #     print('Cross-validation:',count,', Elapsed time is %s seconds' % (time.time() - tic))
    #     print('base acc', base)
    #     # Plotting
    #     # if plot_or_not:
    #     #     fig, ax = plt.subplots(figsize=[14,5])
    #     #     ax.plot(loss_t, 'g-', label = 'training loss')
    #     #     ax.plot(loss_test, 'r-', label = 'test loss')
    #     #
    #     #     ax.set_ylabel('Cross Entropy Loss')
    #     #     ax.legend()
    #     #     ax.set_xlabel('epochs')
    #     #     ax.set_title(" Hidden Layers = " + str(current_setup.numHlayers) +\
    #     #                   " Hidden units = " + str(current_setup.numUnits) + " Dropout = " + str(current_setup.dropout_rate) +\
    #     #                   " Weight decay = " + '{:.4f}'.format(current_setup.wd))
    #     #     plt.show()
    #
    #
    # Training_time = round((time.time() - tic), 2)
    # for cv_num in range(len(accuracy)):
    #     Results = pd.concat([Results, pd.DataFrame(
    #         {'Model': [name], 'Fold': ['Fold' + str(cv_num + 1)], 'Accuracy': [accuracy[cv_num]],
    #          'Computer_info': [computer_info], 'n_jobs': [n_jobs],
    #          'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)
    # # save in every iteration
    # avg_acc = np.mean(accuracy)
    # Results = pd.concat([Results, pd.DataFrame(
    #     {'Model': [name], 'Fold': ['Average'], 'Accuracy': [avg_acc], 'Computer_info': [computer_info],
    #      'n_jobs': [n_jobs],
    #      'base': [base], 'Run_time_5CV_second': [Training_time]})], sort=False)
    #
    # # save in every iteration

    Results.to_csv(output_file_path,index=False)




if __name__ == '__main__':


    tic = time.time()
    # Parameters:


    sample_size_list = ['10k']
    # sample_size_list = ['10k']

    computer_info = 'I9-9900K'

    Re_run = True # True: rerun all models, False: if there are results existed, jump it

    # Dependent_var_list = ['MODE','CAR_OWN','TRIPPURP']
    Dependent_var_list = ['MODE']
    verbose_or_not = False
    dnn_struc_list = [(1,30),(3,30),(5,30),
                      (1,100),(3,100),(5,100),
                      (1,200),(3,200),(5,200)]

    # dnn_struc_list = [(5,200)]
    # DATASET_list = ['NHTS', 'London', 'SG']
    DATASET_list = ['London']


    col_cost = 'cost_driving_fuel'
    num_points = 100

    MODE_WANT = 4 # drive


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

                Estimator_name = 'DNN_GradientBoost' + '_' + str(numUnits) + '_' + str(numHlayers) + '_python'

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
                    print('Read raw data time:', round(time.time() - tic, 1),'s')
                    print(data.values.shape)
                    value = data[Dependent_var].value_counts()
                    base = value.max() / len(data)
                    print('Read raw data time:', round(time.time() - tic, 1), 's')
                    x_train_nn_all = np.array(data.iloc[:, 1:])


                    num_cross_validation = 5
                    choice_train_nn_all = np.array(data.iloc[:, 0])
                    DNN_estimate(x_train_nn_all, data_test, col_cost, num_points, MODE_WANT,  choice_train_nn_all, num_cross_validation, base, output_file_path, DNN_para, Re_run, computer_info,
                                 verbose_or_not,training_epochs,n_estimator,Estimator_name, n_jobs=1)
                    print('Elapsed time is %s seconds' % (time.time() - tic))




