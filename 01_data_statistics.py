import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Count_info(data, ref_table):
    num_trips = len(data)
    print ('Number of trips', len(data))
    data_individual = data.drop_duplicates(["HOUSEID", "PERSONID"])
    print ('Number of individual', len(data_individual))
    num_individual = len(data_individual)
    print ('Number of household', len(pd.unique(data['HOUSEID'])))
    num_household = len(pd.unique(data['HOUSEID']))

    # Add number of input variables of different dependent varibles

    #print ('Number of independent variables', len(list(ref_table.loc[ref_table['MNL_input'] == 1,'Input_variables'])))

    data_save = pd.DataFrame({'Num_trips':[num_trips],'Num_individual':[num_individual],'Num_household':[num_household]})
    data_save.to_csv('output/NHTS_statistics.csv',index=False)






def dependent_variables_100k(data, task, save_fig):

    colors = ["black", "gray"]
    font_size = 20

    if task == 'NHTS-MC':
        # mode choice
        # 1 walk and bike 2 car 3 suv 4 van and truck 5 public transit, 6 other
        columns = [1,2,3,5,4,6]
        density = []
        total = 0
        for key in columns:
            num = len(data.loc[data['MODE'] == key])
            total += num
            density.append(num/ len(data))
        one = sum(density)
        labels_list = ['WB', 'Car', 'SUV', 'PT','VT', 'Others']
        x_label = 'Mode choice'
        rotation = 0

    elif task == 'NHTS-TP':
        #
        columns = [4,1,2,3,5]
        # 1 Home based other 2 home based shop 3 home based social 4 home based work 5 non home based
        density = []
        total = 0
        for key in columns:
            num = len(data.loc[data['TRIPPURP'] == key])
            total += num
            density.append(num/ len(data))
        labels_list = ['HB work','HB other', 'HB shop', 'HB social',  'Non HB']
        x_label = 'Trip purpose'
        rotation = 0
    elif task == 'NHTS-CO':
        #
        columns = [1,2,3,4,5]
        # 1 carown = 0;
        # 2 carown = 1 ...
        # 3 carown = 2 ...
        # 4 carown = 3 ...
        # 5:carown>=4
        density = []
        total = 0
        for key in columns:
            num = len(data['CAR_OWN'].loc[data['CAR_OWN']==key])
            total += num
            density.append(num/ len(data))
        labels_list = ['0', '1', '2', '3', '>3']
        x_label = 'Household car ownership'
        rotation = 0
    elif task == 'LTDS-MC':
        # mode choice
        # 1: walk, 2: cycle, 3: public transport, 4:drive)
        columns = [1,2,3,4]
        density = []
        total = 0
        for key in columns:
            num = len(data.loc[data['MODE'] == key])
            total += num
            density.append(num/ len(data))
        one = sum(density)
        labels_list = ['Walk', 'Cycle', 'PT', 'Drive']
        x_label = 'Mode choice'
        rotation = 0
    else:
        columns = [1,2,3,4,5]
        # key_choice_index = {'Walk': 1, 'PT': 2, 'RH': 3, 'AV': 5, 'Drive': 4}
        density = []
        total = 0
        for key in columns:
            num = len(data.loc[data['MODE'] == key])
            total += num
            density.append(num/ len(data))
        one = sum(density)
        labels_list = ['Walk', 'PT',  'Rail hailing', 'Drive', 'AV']
        x_label = 'Mode choice'
        rotation = 0


    def plot_dsn(density, x_label, labels_list, y_label, save_fig, rotation):

        N = len(labels_list)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.5      # the width of the bars
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(ind, density, width, color=colors[0])
        # rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
        # print ('----------------')
        plt.yticks(fontsize=font_size)
        #plt.ylim(0.35,0.46)
        #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
        ax.set_ylabel(y_label,fontsize=font_size)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels_list,fontsize=font_size, rotation = rotation)
        ax.set_xlabel(x_label,fontsize=font_size)
        # ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
        plt.tight_layout()



    plot_dsn(density,x_label,labels_list,'Density', save_fig, rotation)
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + task + '.png', dpi=300)

def dependent_variables_dsn(data):


    colors = ["black", "gray"]

    def plot_dsn(density, x_label, labels_list, y_label, save_fig, rotation):

        N = len(labels_list)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.5      # the width of the bars
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(ind, density, width, color=colors[0])
        # rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
        # print ('----------------')
        plt.yticks(fontsize=16)
        #plt.ylim(0.35,0.46)
        #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
        ax.set_ylabel(y_label,fontsize=16)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels_list,fontsize=16, rotation = rotation)
        ax.set_xlabel(x_label,fontsize=16)
        # ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            plt.savefig('img/'+ y_label +'.png', dpi=300)
    # mode choice
    columns = [[1],[3],[4],[5,6]]
    density = []
    total = 0
    for key in columns:
        num = len(data['TRPTRANS'].loc[data['TRPTRANS'].isin(key)])
        total += num
        density.append(num/ len(data))
    density.append((len(data)-total) / len(data))
    labels_list = ['Walk&Bike','Car','SUV','Van & Truck','Other']
    plot_dsn(density,'Mode Choice',labels_list,'Density', 0, 0)
    # Trip purpose

    columns = ['TRIPPURP_HBW','TRIPPURP_HBO','TRIPPURP_HBSHOP','TRIPPURP_HBSOCREC','TRIPPURP_NHB']
    density = []
    for key in columns:
        num = len(data['TRIPPURP'].loc[data['TRIPPURP'] == key])
        density.append(num/ len(data))
    labels_list = ['Home-based work', 'Home-based shop', 'Home-based social', 'Home-based other', 'Non home-based']
    plot_dsn(density, 'Trip purpose', labels_list, 'Density', 0, 60)
    # Car ownership
    columns = [0,1,2,3]
    density = []
    for key in columns:
        num = len(data['HHVEHCNT'].loc[data['HHVEHCNT'] == key])
        density.append(num/ len(data))
    num = len(data['HHVEHCNT'].loc[data['HHVEHCNT'] >= 4])
    density.append(num / len(data))
    labels_list = ['0', '1', '2', '3','>=4']
    plot_dsn(density, 'Car ownership', labels_list, 'Density',0, 0)
    a=1

if __name__ == '__main__':
    tic = time.time()
    # data = pd.read_csv('data/data_input_V1.csv')
    # ref_table = pd.read_csv('data/Input_variables.csv')
    print('Read raw data time:', round(time.time() - tic, 1),'s')
    # Count_info(data, ref_table)

    #####################################
    ##############
    save_fig = 1
    task_list = ['NHTS-MC', 'NHTS-TP','NHTS-CO','LTDS-MC','SGP-MC']
    file_list = ['data/data_mode_choice_100k.csv','data/data_trip_purpose_100k.csv','data/data_car_ownership_100k.csv',
                 'London_dataset/data_London_mode_choice_100k.csv','SG_dataset/data_SG_mode_choice_10k.csv']
    for task, file_dir in zip(task_list, file_list):
        # task = 'SGP-MC'
        # file_dir = 'SG_dataset/data_SG_mode_choice_10k.csv'
        data_100k = pd.read_csv(file_dir)
        ##############
        dependent_variables_100k(data_100k, task, save_fig)

    #####################################
    # Whether normalize data
    # m.vif_test()
    # m.MNL_TEST()
    # m.excute()