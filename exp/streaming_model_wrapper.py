import numpy as np
import math
from models.streaming.base_model_adapter import AdapterTSBAD, PostProcessedModel, AdapterDSalmon

Point_wize_AD_Pool = [ 'LODA', 'RSHash', 'RRCF', 'HSTree', 'MemStream', 'MCOD', 'SDOstream', 'xStream']#, 'xStreamPysad'] 
Normalized_dSalmon_Pool = ['SWKNN']


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


def fit_LODA(data_train):
    from models.streaming.LODA import LODAopt
    clf = AdapterTSBAD(LODAopt())
    clf.fit(data_train)
    return clf

def fit_RRCF(data_train, num_trees = 4, shingle_size = 4, tree_size = 256):
    from pysad.models import RobustRandomCutForest    
    clf = AdapterTSBAD(RobustRandomCutForest(num_trees = num_trees, shingle_size = shingle_size, tree_size = tree_size))
    clf.fit(data_train)
    return clf

def fit_HSTree(data_train, window_size = 100, num_trees = 25, max_depth = 15):
    from pysad.models import HalfSpaceTrees
    clf = AdapterTSBAD(HalfSpaceTrees(feature_mins= np.min(data_train, axis = 0), feature_maxes= np.max(data_train, axis = 0), initial_window_X = data_train,
                                      window_size = window_size, num_trees = num_trees, max_depth = max_depth))
    #clf.fit(data_train)
    return clf

def flip_sign(s):
    # Préserve le type d'origine
    if np.isscalar(s):  # float, int, np.float64
        return -s
    else:
        s = np.asarray(s)
        return -s

def fit_RSHash(data_train, sampling_points = 1000, decay = 0.015, num_components = 100, num_hash_fns = 1):
    from pysad.models import RSHash
    
    base = RSHash(
        feature_mins=np.min(data_train, axis=0),
        feature_maxes=np.max(data_train, axis=0),
        sampling_points=sampling_points,
        decay=decay,
        num_components=num_components,
        num_hash_fns=num_hash_fns
    )

    
    wrapped = PostProcessedModel(base_model=base, transform=flip_sign)
    clf = AdapterTSBAD(wrapped)
    clf.fit(data_train)
    return clf

def fit_MemStream(data_train, memory_len = 256, beta = 0.1):
    from models.streaming.MemStream import MemStream
    in_dim = data_train.shape[1]
    clf = MemStream(in_dim=in_dim ,memory_len=memory_len, beta=beta)
    clf.fit(data_train)
    return clf


def fit_LEAP(data_train, k=5, R=1.0, slidingWindow=100, slide=10):
    from models.streaming.LEAP import LEAP
#    jar_path="models/streaming_models/jar_files/leap-1.0.0-jar-with-dependencies.jar"
    jar_path="/home/d66285/These/TSBAD_Streaming/online_top_k_TSBAD/models/streaming_models/LEAP/target/leap-1.0.0-jar-with-dependencies.jar"
    clf = LEAP(k=k, R=R, W=slidingWindow, slide=slide, jar_path= jar_path, jvm_opts=None)
    clf.fit(data_train)
    return clf

def fit_MCOD(data_train, k=5, R=1.0, W=100, slide=10):
    from models.streaming.MCOD import MCOD
    jar_path="/home/d66285/These/TSBAD_Streaming/online_top_k_TSBAD/models/streaming_models/jar_files/mcod-1.0.0-jar-with-dependencies.jar"
    clf = MCOD(k=k, R=R, W=W, jar_path= jar_path)
    clf.fit(data_train)
    return clf

def fit_SWKNN(data_train, slidingWindow = 100, k = 5):
    from outlier import SWKNN
    clf = AdapterDSalmon(SWKNN(window = slidingWindow, k = k, k_is_max = False)) 
    return clf

# def fit_xStream(data_train, window= 10, n_estimators= 10,  n_projections= 10, depth= 10):
#     from outlier import xStream
#     clf = AdapterDSalmon(xStream(window = window, n_estimators=n_estimators, n_projections=n_projections, depth=depth))
#     clf.set_initial_sample(data_train)
#     clf.fit(data_train)
#     return clf

def fit_xStream(data_train, window= 10, n_estimators= 10,  n_projections= 10, depth= 10):
    from outlier import xStream
    clf = AdapterDSalmon(xStream(window = window, n_estimators=n_estimators, n_projections=n_projections, depth=depth))
    if window < data_train.shape[0]:
        init_data = data_train[:window,:]
        clf.set_initial_sample(init_data)
    return clf


# def fit_xStreamPysad(data_train, window = 256, n_chains= 200, n_components= 50, depth= 20):
#     from pysad.models import xStream
#     clf = AdapterTSBAD(xStream(window_size = window, n_chains= n_chains, num_components= n_components, depth= depth))
#     #clf.fit(data_train)
#     return clf

# def fit_xStream(data_train, **kwargs):
#     try:
#         from outlier import xStream

#         slidingWindow = kwargs.get('slidingWindow', 10)
#         n_estimators = kwargs.get('n_estimators', 10)
#         n_projections = kwargs.get('n_projections', 10)
#         depth = kwargs.get('depth', 10)
#         print(f"slidingWindow: {slidingWindow}, n_estimators: {n_estimators}, n_projections: {n_projections}, depth: {depth}")

#         clf = AdapterDSalmon(xStream(window=slidingWindow, n_estimators=n_estimators, n_projections=n_projections, depth=depth))
#         clf.set_initial_sample(data_train)
#         clf.fit(data_train)
#         return clf
#     except Exception as e:
#         import traceback
#         print("Traceback:")
#         traceback.print_exc()
#         raise e


def fit_SDOstream(data_train, k=200, T=1000, x = 5):
    from outlier import SDOstream
    clf = AdapterDSalmon(SDOstream(k=k, T=T, qv = 0.3, x = x))
    return clf


# def fit_RSHash(data_train, sampling_points = 1000, decay = 0.015, num_components = 100, num_hash_fns = 1):
#     from pysad.models import RSHash
#     clf  = AdapterTSBAD(RSHash(feature_mins= np.min(data_train, axis = 0), feature_maxes= np.max(data_train, axis = 0),
#                                sampling_points = sampling_points, decay = decay, num_components = num_components, num_hash_fns = num_hash_fns))
#     clf.fit(data_train)
#     return clf
