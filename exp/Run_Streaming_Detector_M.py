# -*- coding: utf-8 -*-
# Adapted from Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging, sys
#sys.path.insert(0, r"/home/d66285/Desktop/Code/pyStreamAD")
#sys.path.append(r"/home/d66285/Desktop/Code/TSBAD")
# sys.path.insert(0, r"C:/Users/d66285/Desktop/Code/streaming_methods_pysad/pyStreamAD")
# sys.path.append(r"C:/Users/d66285/Desktop/Code/TSBAD")
#sys.path.insert(0, r"/home/mparrino/Desktop/Code/pyStreamAD")
#sys.path.append(r"/home/mparrino/Desktop/Code/TSBAD")
#sys.path.append(r"/home/mparrino/Desktop/Code/dSalmon/python")
#print(sys.path)
from streaming_model_wrapper import *
from ..HP_list import Optimal_Stream_algo_HP_dict
from ..utils.stream_analysis import normalize_minmax_score
from ..utils.utility import ZNormalizer

# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from ..models.online.feature import Window, Window3D
# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())


def compute_metrics(filename, write_csv, final_run_time,final_output, label, slidingWindow):
    try:
        evaluation_result = get_metrics(final_output, label, slidingWindow=slidingWindow) #slidingWindow for VUS
        print('evaluation_result: ', evaluation_result)
        list_w = list(evaluation_result.values())
    except:
        list_w = [0]*9
    list_w.insert(0, final_run_time.sum())
    list_w.insert(0, filename)
    write_csv.append(list_w)

    ## Temp Save
    col_w = list(evaluation_result.keys())
    col_w.insert(0, 'Time')
    col_w.insert(0, 'file')
    w_csv = pd.DataFrame(write_csv, columns=col_w)            

    return w_csv            



if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='../TSBAD/Datasets/TSB-AD-M/')
    parser.add_argument('--file_list', type=str, default='../TSBAD/Datasets/File_List/TSB-AD-M-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/streaming/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/streaming/')
    parser.add_argument('--time_dir', type=str, default='eval/time/streaming/')

    parser.add_argument('--save', type=bool, default=True)  # To modify for the real launch to True 
    parser.add_argument('--AD_Name', type=str, default='RRCF')
    args = parser.parse_args()


    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    target_time_dir = os.path.join(args.time_dir, args.AD_Name)
    os.makedirs(target_time_dir, exist_ok = True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}_online.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_list)['file_name'].values
 
    Optimal_Det_HP = Optimal_Stream_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)


    csv_path = f'{args.save_dir}/{args.AD_Name}.csv'
    os.makedirs(args.save_dir, exist_ok=True)
    
    processed_files = set()

    if os.path.exists(csv_path):
        df_metrics = pd.read_csv(csv_path)
        processed_files = set(df_metrics['file'].values)

    write_csv = []
    for filename in [file_list[153]]:
        score_file = os.path.join(target_dir, filename.split('.')[0] + '.npy')
        score_exists = os.path.exists(score_file)
        metric_exists = filename in processed_files

        if score_exists and metric_exists:
            continue

        print('Processing:{} by {}'.format(filename, args.AD_Name))
        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        feats = data.shape[1]
        
        train_index = filename.split('.')[0].split('_')[-3]
        train_index = int(train_index)
        data_train = data[:train_index, :]
        print("mean = ", np.mean(data_train, axis=0), " and std = ", np.std(data_train, axis=0))

        if args.AD_Name in Normalized_dSalmon_Pool:
            standardizer = ZNormalizer()
            data_train = standardizer.fit_transform(data_train)
            data = standardizer.transform(data)

            print("mean = ", np.mean(data_train, axis=0), " and std = ", np.std(data_train, axis=0))

        slidingWindow = 1
        if 'slidingWindow' in Optimal_Det_HP:
            slidingWindow = Optimal_Det_HP['slidingWindow']
            print(f'Sliding window = {slidingWindow}, data.shape = {data.shape}')
            data_stream = Window3D(window = slidingWindow).convert(data) 
            print(f'data_stream.shape = {data_stream.shape}')
        else:
            print(f'Sliding window = {slidingWindow}, data.shape = {data.shape}')
            data_stream = Window(window = slidingWindow).convert(data) 
            print(f'data_stream.shape = {data_stream.shape}')
        vus_slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)


        start_time = time.time()
        clf= fit_AD(args.AD_Name, data_train, **Optimal_Det_HP)
        end_time = time.time()
        
        training_time = end_time - start_time

        final_output = np.zeros(len(data))
        
        if args.AD_Name in Point_wize_AD_Pool:
            final_run_time = np.zeros(len(data) +1)
        else:
            final_run_time = np.zeros(len(data[slidingWindow-2:]))

        if data_stream.shape[0] != len(final_run_time[1:]):
            raise Exception(f'pas de cohérence entre nombre de fenêtres: {data_stream.shape[0]} et nb de run: {len(final_run_time[1:])}')
        
        final_run_time[0] = training_time

        errors = 0 #regarder data_stream.shape
        for i, data_window in enumerate(data_stream):
            start_time = time.time()
            output = clf.decision_function(data_window)
            end_time = time.time()
            run_time = end_time - start_time

            if args.AD_Name == "LEAP":                
                if isinstance(output, np.ndarray) and output.size > 0:
                    slide = Optimal_Det_HP['slide']
                    if output.size != slide:
                        logging.error(f"LEAP: score size {output.size} != slide={slide}")
                        errors += 1
                    else:
                        start_idx_fin = (slidingWindow - slide) + i
                        end_idx_fin   = (slidingWindow - 1) + i
                        final_output[start_idx_fin : end_idx_fin + 1] = output
                        #print(f"[LEAP write] [{start_idx_fin}:{end_idx_fin}] <- {output}")

            else:
                if isinstance(output, np.ndarray):    

                    final_output[slidingWindow -1 + i]= output[-1]  
                elif isinstance(output, float): 
                    final_output[slidingWindow -1 + i]= output
                else:
                    logging.error(f'At {filename}, sliding window number {i}: '+output)
                    errors+=1

            final_run_time[i+1] = run_time

        # Padding scores
        final_output[: slidingWindow-1] = final_output[slidingWindow] 
        if args.AD_Name =="xStream":
            final_output[:Optimal_Det_HP['window']] = final_output[Optimal_Det_HP['window']]
        normalized_output = normalize_minmax_score(final_output)

        if errors ==0:
            logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {final_run_time.sum():.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', final_output)

        ### whether to save the evaluation result
        if args.save:

            # w_csv = compute_metrics(final_output, label, vus_slidingWindow)
            
            # os.makedirs(args.save_dir, exist_ok=True)           
            # w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            new_row = compute_metrics(filename, [], final_run_time, final_output, label, vus_slidingWindow)
            new_row.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            np.save(os.path.join(target_time_dir, filename.split('.')[0] + '.npy'), final_run_time)

            normalized_target_dir = target_dir.replace('streaming', 'streaming_minmax')
            os.makedirs(normalized_target_dir, exist_ok = True)
            np.save(normalized_target_dir+'/'+filename.split('.')[0]+'.npy', normalized_output)

    logging.shutdown()






            