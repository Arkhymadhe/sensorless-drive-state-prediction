from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


def import_model(max_depth = 3, n_jobs = -1, random_state = 42):
    ''' Return base model. '''
    
    return ExtraTreesClassifier(max_depth = max_depth, n_jobs = n_jobs, random_state = random_state)


def wrap_model(base_classifier, random_state = 42):
    ''' Wrap model as a Base model for boosting machine. '''
    
    return AdaBoostClassifier(base_estimator = base_classifier, random_state = random_state)


def train_model(model, X, y):
    ''' Fit model to dataset. '''
    
    model.fit(X, y)
    
    return model