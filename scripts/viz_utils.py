import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_confusion_matrix

sns.set()

# Univariate visualization
def univariate_plot(data, path, save = True):
    
    ''' Plot the data univariately. '''
    
    for col in data.columns:
        plt.figure(figsize = (10, 8))
        sns.displot(data[col])

        plt.title(f'Distribution plot for Feature {col}')
        
        if save:
            plt.savefig(f'{path} - Feature {col}.png', dpi = 300)
        
        plt.show()
        plt.close('all')
        
    return None


def correlogram(data, path, h = 10, w = 10, save = True):
    ''' Plot and save correlogram. '''
    
    plt.figure(figsize = (h, w))
    sns.pairplot(data = data, hue = 48)
    
    if save:
        plt.savefig(f'{path}.png', dpi = 300)
    
    plt.show()
    plt.close('all')
    
    return None


def visualize_confusion_matrix(model, X, y, split, path, save = True):
    """ Display Confusion Matrix visually."""

    plot_confusion_matrix(model, X, y)
    if save:
        plt.savefig(os.path.join(path, f'{split}-confusion-matrix.png'), dpi = 300)
    plt.show()
    plt.close('all')

    return None