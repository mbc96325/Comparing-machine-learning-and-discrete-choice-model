
import pandas as pd




def num_dnn(Results_all):
    Results_all_dnn = Results_all.loc[Results_all['Clf_cate']=='DNN']
    print('total sample dnn', len(Results_all_dnn))
    num_dnn = pd.unique(Results_all_dnn['Model'])
    print('num_dnn', len(num_dnn))

def num_models(Results_all):
    all_big_cate = list(pd.unique(Results_all['Clf_cate']))
    for cate in all_big_cate:
        print(cate, len(pd.unique(Results_all.loc[Results_all['Clf_cate']==cate, 'Model'])))

if __name__ == '__main__':
    # generate_date_size()
    Results_all = pd.read_csv('output/All_results.csv')
    # print(len(pd.unique(Results_all['Model'])))

    num_models(Results_all)

    num_dnn(Results_all)