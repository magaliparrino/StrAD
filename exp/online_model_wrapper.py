import numpy as np
import math
from utils.multiSlidingWindows import find_length_rank_for_multi

# Unsupervise_AD_Pool = ['Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'Sub_PCA', 'PCA', 'HBOS', 'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 
#                        'CBLOF', 'EIF', 'RobustPCA']
# Semisupervise_AD_Pool = ['MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly',  'AnomalyTransformer', 'TimesNet', 'FITS', 
#                          'Donut', 'OFA']

Optimized_windowSize_HP_Pool = ['CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly',  'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 
                                'OFA', 'KMeansAD']
Non_optimized_windowSize_HP_Pool = ['Sub_IForest', 'Sub_LOF', 'Sub_PCA', 'Sub_HBOS', 'Sub_KNN', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 
                                    'Sub_MCD', 'Sub_OCSVM', 'AutoEncoder']
Point_wize_AD_Pool = [ 'LOF', 'HBOS', 'KNN', 'MCD', 'OCSVM'] 

Last_index_pred = ['CNN', 'LSTMAD', 'Donut', 'FITS', 'TimesNet', 'TranAD', 'AnomalyTransformer']


def fit_AD(model_name, data_train, **kwargs):
    try:
        function_name = f'fit_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message
    

def fit_IForest(data_train, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from ..models.online.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data_train)
    return clf

def fit_LOF(data_train, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from ..models.online.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data_train)
    return clf

def fit_PCA(data_train, slidingWindow=100, n_components=None, n_jobs=1):
    from ..models.online.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data_train)
    return clf

def fit_HBOS(data_train, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from ..models.online.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data_train)
    return clf

def fit_OCSVM(data_train, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from ..models.online.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    return clf

def fit_MCD(data_train, support_fraction=None, slidingWindow=1, n_jobs=1):
    from ..models.online.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    return clf

def fit_KNN(data_train, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from ..models.online.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data_train)
    return clf

def fit_KMeansAD(data_train, n_clusters=20, window_size=20, n_jobs=1):
    from ..models.online.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    clf.fit(data_train)
    return clf

def fit_CBLOF(data_train, n_clusters=8, alpha=0.9, n_jobs=1):
    from ..models.online.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data_train)
    return clf

def fit_RobustPCA(data_train, max_iter=1000):
    from ..models.online.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data_train)
    return clf

def fit_AutoEncoder(data_train, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    from ..models.online.AE import AutoEncoder
    slidingWindow = find_length_rank_for_multi(data_train, rank=1)
    clf = AutoEncoder(slidingWindow=slidingWindow, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    return clf

def fit_CNN(data_train, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):
    from ..models.online.CNN import CNN
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_train.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    return clf

def fit_LSTMAD(data_train, window_size=100, lr=0.0008):
    from ..models.online.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_train.shape[1], batch_size=128)
    clf.fit(data_train)
    return clf

def fit_TranAD(data_train, win_size=10, lr=1e-3):
    from ..models.online.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_train.shape[1], lr=lr)
    clf.fit(data_train)
    return clf

def fit_AnomalyTransformer(data_train, win_size=100, lr=1e-4, batch_size=128):
    from ..models.online.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_train.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    return clf

def fit_OmniAnomaly(data_train, win_size=100, lr=0.002):
    from ..models.online.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_train.shape[1], lr=lr)
    clf.fit(data_train)
    return clf

def fit_USAD(data_train, win_size=5, lr=1e-4):
    from ..models.online.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_train.shape[1], lr=lr)
    clf.fit(data_train)
    return clf

def fit_TimesNet(data_train, win_size=96, lr=1e-4):
    from models.online.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_train.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    return clf

def fit_FITS(data_train, win_size=100, lr=1e-3):
    from ..models.online.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_train.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    return clf

