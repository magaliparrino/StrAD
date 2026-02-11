
Stream_algo_HP_dict = {
    'RRCF': {
        "num_trees": [ 2, 4, 8, 16, 20, 25],       
        "shingle_size": [2, 4, 8],            
        "tree_size": [256, 512]
    },

    'RSHash': {
        "sampling_points": [500, 1000],
        "decay": [0.01, 0.015, 0.02],
        "num_components": [50, 100, 200, 300],
        "num_hash_fns": [1, 2, 4]
    },
    'HSTree': {
        "window_size": [50, 100, 200, 250],  
        "num_trees": [10, 25, 50, 100],      
        "max_depth": [10, 15]               
    },
    'MemStream': {
        'memory_len':{32, 64, 256, 512, 1024},
        'beta':{10,1,0.1, 0.01,0.001},
    },
    'LEAP': {
        'k': {5, 10, 20, 40},
        'R': {0.5, 1.0, 2.0},
        'slidingWindow': {256, 512},
        'slide':{32, 64},
    },
    'MCOD': {
        'k': {5, 10, 20, 40},
        'R': {0.5, 1.0, 2.0},
        'W':{64, 256, 512, 1024},
    },
    'xStream': {
        'window':{64, 128, 256},
        'n_estimators': {50,100,200},
        'n_projections':{50,100,200},
        'depth':{10,15,20},
    },
    'SWKNN': {
        'slidingWindow': {64,128,256,512,1024},
        'k':{5,10,15,20,30,40},
    },
    'SDOstream': {
        'k':{200,500,1000,1500},
        'T':{128,256,512},
        'x':{3,6,10}
    }

}

Optimal_Stream_algo_HP_dict = {
    'HSTree': {'window_size': 200, 'num_trees': 25, 'max_depth': 15},
    'RSHash': {'sampling_points': 1000,
               'decay': 0.01,
               'num_components': 300,
               'num_hash_fns': 2},    
    'LODA' : {},
    'RRCF': {'num_trees': 16, 'shingle_size': 8, 'tree_size': 512},  
    'MemStream' : { 'memory_len' : 32, 'beta' : 0.001},
    'LEAP': { 'k':40, 'R':2.0, 'slidingWindow':256, 'slide':32},
    'MCOD':{ 'k':5, 'R':1.0, 'W':1024},
    'SWKNN': {'slidingWindow': 64, 'k': 40},
    'xStream': {'window': 256, 'n_estimators': 200, 'n_projections': 50, 'depth': 20},
    'SDOstream': {'k': 1000, 'T': 512, 'x': 10},
}