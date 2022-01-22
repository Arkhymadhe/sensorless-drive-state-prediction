import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_ops import split_data
from data_ops import variables

from model_utils import train_model


def important_features(model, data, display : bool, path, save : bool = True, sort : bool = True):
    ''' Return feature importances. '''

    data, targets = variables(data)

    X_train, _, y_train, _ = split_data(data, targets, split_size=0.25)
    model = train_model(model, X_train, y_train)
    
    if hasattr(model, 'feature_importances_'):
        feature_ranks = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_ranks = model.coef_
    else:
        raise TypeError('Model has no feature ranking utility.')
    
    if display:
        plt.figure(figsize = (20, 8))
        sns.barplot(x = data.columns, y = 10**3*feature_ranks, color = 'blue')

        plt.title('Feature importances for data features')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        
        if save:
            plt.savefig(os.path.join(path, 'images', '3', 'Feature rankings.png'), dpi = 300)
        plt.show()
        plt.close('all')
        
    importances = pd.DataFrame({'Importance' : 1000*feature_ranks,
                                'Feature' : ['Feature {}'.format(i + 1) for i in data.columns]
                               })
    if save:
        importances.to_csv(os.path.join(path, 'text', 'feature_importance.csv'), index = False)
    
    return importances.sort_values(by = 'Importance') if sort else importances


def get_important_features(data, feature_ranking, threshold = 1):
    ''' Return most important feature subset by threshold. '''
    
    feature_ranking = feature_ranking.sort_values(by = 'Importance')
    
    f_name = feature_ranking.loc[feature_ranking['Importance'] > threshold, 'Feature']
    
    f_list = f_name.apply(lambda f: int(f.split(' ')[1]) - 1)
    
    f_list = list(f_list)
    f_list = sorted(f_list)
    
    data = data.loc[:, f_list]
    
    return data