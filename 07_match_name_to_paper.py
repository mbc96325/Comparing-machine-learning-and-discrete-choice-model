import pandas as pd
import numpy as np



def get_model_name_before_manual():
    name_list = pd.read_csv('output/All_results.csv')
    name_list = name_list.loc[:,['Model','Clf_cate','Programming']].drop_duplicates().reset_index(drop=True)

    file_name = 'output/latex_table.txt'
    col_1 = []
    col_2 = []
    with open(file_name) as fin:
        for line in fin:
            if '&' in line and 'textbf' not in line:
                first_ele = line.split('&')[0]
                first_ele = first_ele.replace(' ','')
                first_ele = first_ele.replace('\\','')
                col_1.append(first_ele)
                second_ele = line.split('&')[1]
                second_ele = second_ele.replace(' ', '')
                col_2.append(second_ele)

    df = pd.DataFrame(np.array([col_1,col_2]).T, columns = ['Model_name_latex','latex_clf_cate'])

    Big_cate_corresponding = {'DCM':'DCM','DNN':'DNN','DA':'DA','BY':'BM','SVM':'SVM','KNN':'KNN','DT':'DT','GLM':'GLM',
                              'GP':'GP','Bagging':'BAGGING','RF':'RF','Boosting':'BOOSTING'}

    name_list['latex_clf_cate'] = name_list['Clf_cate'].apply(lambda x: Big_cate_corresponding[x])
    for key, i in zip(Big_cate_corresponding,range(len(Big_cate_corresponding))):
        name_list.loc[name_list['latex_clf_cate']==Big_cate_corresponding[key], 'cate_id'] = i

    name_list = name_list.merge(df, on = ['latex_clf_cate'],how = 'left')
    name_list = name_list.sort_values(['cate_id','Model'])
    name_list.to_csv('output/model_name_list_code_out_need_manually_process.csv',index=False)

def check_manual_result():
    ##################+========================check_manually_results

    name_list = pd.read_csv('output/model_name_list_code_out_manually_processed.csv')
    print('num models', len(name_list))
    print('num cate', len(name_list.drop_duplicates(['Model_name_latex'])))

    name_list_dup =name_list.loc[name_list.duplicated(['Model_name_latex'])]
    print(name_list_dup)


    file_name = 'output/latex_table_new.txt'
    col_1 = []
    col_2 = []
    with open(file_name) as fin:
        for line in fin:
            if '&' in line and 'textbf' not in line:
                first_ele = line.split('&')[0]
                first_ele = first_ele.replace(' ','')
                first_ele = first_ele.replace('\\','')
                col_1.append(first_ele)
                second_ele = line.split('&')[1]
                second_ele = second_ele.replace(' ', '')
                col_2.append(second_ele)

    df = pd.DataFrame(np.array([col_1,col_2]).T, columns = ['Model_name_latex','latex_clf_cate'])

    latex_num = len(df)
    print('num of model in latex',len(df))
    print('num of model cate in latex',len(df.drop_duplicates(['latex_clf_cate'])))

    if latex_num < len(name_list):
        name_list = name_list.merge(df, on = ['Model_name_latex'], how = 'left')
        print('no data latex', name_list.loc[name_list['latex_clf_cate_y'].isna()])
    if latex_num > len(name_list):
        df = df.merge(name_list, on = ['Model_name_latex'], how = 'left')
        print('need to drop data latex', df.loc[df['latex_clf_cate_y'].isna()])
    a=1

if __name__ == '__main__':
    check_manual_result()