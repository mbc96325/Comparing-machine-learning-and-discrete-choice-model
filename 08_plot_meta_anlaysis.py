import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    file_dir = '../literature_review/literature_review.csv'

    df = pd.read_csv(file_dir)


    # Remove FL
    df = df.loc[df['author'] != 'Pulugurta', :]
    df = df.loc[df['author'] != 'Tai', :]
    df = df.loc[df['author'] != 'Thill', :]
    df = df.loc[df['author'] != 'Shmueli', :]



    years, authors = list(df['year']), list(df['author'])
    year_author_unique = sorted(list(set([str(years[i]) + ' ' + authors[i] for i in range(len(years))])))
    NUM_STUDIES = len(year_author_unique)
    TOTAL_MODELS = len(years)
    print("Number of studies: " + str(NUM_STUDIES))
    print('Total models: ' + str(TOTAL_MODELS))
    print('Average number of models per study: ' + str(len(years) / len(year_author_unique)))


    agg_funcs = {'paper_name': 'first',
                 'journal': 'first',
                 'task': 'first',
                 'location': 'first',
                 'num_of_alternative': 'mean',
                 'input_dim': 'mean',
                 'sample_size': 'max',
                 'model': 'count',
                 'best_model': 'first',
                 'model_pred_accuracy': 'max'}
    df.replace([-1, -1.0, '-1', '-1.0'], np.nan, inplace=True)
    grouped = df.groupby(['year', 'author']).agg(agg_funcs)
    grouped.columns = ['paper_name', 'journal', 'task', 'location', 'avg_num_of_alternative', 'avg_input_dim', 'max_sample_size', 'num_models', 'best_model', 'best_model_pred_accuracy']


    print('Average prediction accuracy: ' + str(df['model_pred_accuracy'].mean()))
    print('Min prediction accuracy: ' + str(df['model_pred_accuracy'].min()))
    print('Median prediction accuracy: ' + str(df['model_pred_accuracy'].median()))
    print('Max prediction accuracy: ' + str(df['model_pred_accuracy'].max()))
    print('STD prediction accuracy: ' + str(df['model_pred_accuracy'].std()))


    print('Average BEST prediction accuracy: ' + str(grouped['best_model_pred_accuracy'].mean()))
    print('Min BEST prediction accuracy: ' + str(grouped['best_model_pred_accuracy'].min()))
    print('Median BEST prediction accuracy: ' + str(grouped['best_model_pred_accuracy'].median()))
    print('Max BEST prediction accuracy: ' + str(grouped['best_model_pred_accuracy'].max()))
    print('STD BEST prediction accuracy: ' + str(grouped['best_model_pred_accuracy'].std()))

    # grouped['best_model'].value_counts().plot(kind='bar', title='Best Models', figsize =(10,5))
    # plt.savefig('meta_best_models.png')
    print(grouped['best_model'].value_counts())
    print(np.sum(grouped['best_model'].value_counts()))


    models = df['model'].unique()
    model_acc_df = pd.DataFrame()
    for i, model_type in enumerate(models):
        if i == 0:
            model_acc_df = df.loc[df['model'] == model_type]['model_pred_accuracy'].reset_index()['model_pred_accuracy']
        else:
            model_acc_df = pd.concat([model_acc_df, df.loc[df['model'] == model_type]['model_pred_accuracy'].reset_index()['model_pred_accuracy']], axis=1)

    model_acc_df.columns = models
    return model_acc_df, grouped

def plot_model_family_acc(indicator, model_acc_df,save_fig):

    all_model_family = list(model_acc_df.columns)
    data_df = {'Clf_cate': [], 'Accuracy':[]}
    for key in all_model_family:
        all_sample = list(model_acc_df[key].dropna())
        data_df['Accuracy'] += all_sample
        data_df['Clf_cate'] += [key] * len(all_sample)
    Results_all = pd.DataFrame(data_df)
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
        sns.violinplot(x=Results_all['Clf_cate'], y=Results_all[indicator], color="gray",cut=0)
    sns.stripplot(x = Results_all['Clf_cate'], y = Results_all[indicator],color = 'white',size = 3)
    plt.scatter(x = X, y = Results_all_sinlge['avg_acc'],color = 'blue', label ='Mean',zorder = 5)
    plt.scatter(x=X, y=Results_all_sinlge['median_acc'], color='red', label ='Median',zorder = 4)
    plt.scatter(x=X, y=Results_all_sinlge['max_acc'], color='tab:green', label='Max', zorder=4)
    plt.xlabel('Model families',fontsize=font_size)
    if indicator == 'Accuracy':
        plt.ylabel('Prediction accuracy',fontsize=font_size)
    else:
        plt.ylabel('Training + testing time (log scale)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(X, order_plot, fontsize=font_size, rotation = 90)
    x_lim = [X[0]-0.5, X[-1]+0.5]


    plt.xlim(x_lim[0],x_lim[1])
    if indicator == 'Accuracy':
        y_lim = [0.2,1]
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
                plt.text(x_ - 0.28, 0.3, tx, color='blue', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.3, tx, color='blue', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.3, tx, color='blue', fontsize=font_size)
            count += 1
        count = 0
        for x_, tx in zip(X, median_text):
            if count == 0:
                plt.text(x_ - 0.28, 0.27, tx, color='red', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.27, tx, color='red', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.27, tx, color='red', fontsize=font_size)
            count += 1


        count = 0
        for x_, tx in zip(X, max_text):
            if count == 0:
                plt.text(x_ - 0.28, 0.24, tx, color='tab:green', fontsize=font_size)
            elif count == 1:
                plt.text(x_ - 0.28, 0.24, tx, color='tab:green', fontsize=font_size)
            else:
                plt.text(x_ - 0.28, 0.24, tx, color='tab:green', fontsize=font_size)
            count += 1


    plt.legend(fontsize=font_size,loc='upper right',ncol = 2)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/meta_models_perform_'+ indicator + '.png',dpi = 200)

def plot_bar_best_model_num( grouped, save_fig = 0):
    font_size = 16
    grouped = grouped.reset_index()
    num_best = grouped.groupby(['best_model'])['paper_name'].count().reset_index()
    num_best = num_best.sort_values(['paper_name'],ascending = False).reset_index(drop = True)
    N = len(num_best)
    density = num_best['paper_name']
    ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(ind, density, width, color='k')
    # rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    # print ('----------------')
    plt.yticks(fontsize=font_size)

    x_label = list(num_best['best_model'])
    # plt.ylim(0.35,0.46)
    # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
    ax.set_ylabel('Number of times to be the best model', fontsize=font_size)
    ax.set_xticks(ind)
    rotation = 0
    ax.set_xticklabels(x_label, fontsize=font_size, rotation=rotation)
    ax.set_xlabel('Model families', fontsize=font_size)
    # ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/meta_best_models.png',dpi = 200)
    a=1

if __name__ == '__main__':
    model_acc_df, grouped = load_data()
    plot_model_family_acc(indicator='Accuracy', model_acc_df = model_acc_df,save_fig = 1 )

    # plot_bar_best_model_num(grouped, save_fig = 1)