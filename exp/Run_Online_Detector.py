# -*- coding: utf-8 -*-
# Adapted from Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from TSB_AD.evaluation.metrics import get_metrics
from utils.multiSlidingWindows import find_length_rank_for_multi
from online_model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from ..models.online.feature import Window3D
from utils.stream_analysis import *

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

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

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='../TSBAD/Datasets/TSB-AD-M/')
    parser.add_argument('--file_list', type=str, default='../TSBAD/Datasets/File_List/TSB-AD-M-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/online/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/online/')
    parser.add_argument('--time_dir', type=str, default='eval/time/online/')
    parser.add_argument('--start_from_file', type=int, default=None)

    parser.add_argument('--save', type=bool, default=True)  # To modify for the real launch to True 
    parser.add_argument('--AD_Name', type=str, default='CNN')
    args = parser.parse_args()


    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    target_time_dir = os.path.join(args.time_dir, args.AD_Name)
    os.makedirs(target_time_dir, exist_ok = True)
    logging.basicConfig(filename=f'{target_dir}/000_run_{args.AD_Name}_online.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    file_list = pd.read_csv(args.file_list)['file_name'].values
    if args.start_from_file !=None:
        file_list = file_list[args.start_from_file:]
        count = 0

    # if 'Sub_' in args.AD_Name:
    #     Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name.split('Sub_')[1]]
    # else:
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)

    write_csv = []
    for filename in [file_list[153]]:
        if os.path.exists(target_dir+'/'+filename.split('.')[0]+'.npy'): continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))
        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        
        train_index = filename.split('.')[0].split('_')[-3]
        train_index = int(train_index)
        data_train = data[:train_index, :]

        if args.AD_Name in Non_optimized_windowSize_HP_Pool:
            slidingWindow = find_length_rank_for_multi(data_train, rank=1)
        elif args.AD_Name in Optimized_windowSize_HP_Pool:
            slidingWindow = Optimal_Multi_algo_HP_dict[args.AD_Name][next(iter([param for param in Optimal_Multi_algo_HP_dict[args.AD_Name] if 'win' in param]))]
            if args.AD_Name in ['CNN', 'LSTMAD']:
                slidingWindow+=1
        elif args.AD_Name in Point_wize_AD_Pool:
            slidingWindow = 1
        elif args.AD_Name in ['PCA', 'IForest']:
            slidingWindow = 100
        else:
            raise Exception(f"{args.AD_Name} is not defined")
        
        print(f'Sliding window = {slidingWindow}, data.shape = {data.shape}')
        data_stream = Window3D(window = slidingWindow).convert(data)  #regarder slidingWindow + data.shape
        print(f'data_stream.shape = {data_stream.shape}')


        start_time = time.time()
        clf= fit_AD(args.AD_Name, data_train, **Optimal_Det_HP)
        end_time = time.time()
        # if window != slidingWindow and args.AD_Name not in ['CNN', 'LSTMAD']:
        #     raise Exception(f"{args.AD_Name} has different window for train and for streaming: SlidingWindow stream = {slidingWindow} and train sliding window = {window}")
        # elif window != slidingWindow -1 and args.AD_Name in ['CNN', 'LSTMAD']:
        #     raise Exception(f"{args.AD_Name} has different window for train and for streaming: SlidingWindow stream = {slidingWindow-1} and train sliding window = {window}")
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
        
        #    #testing à virer une fois que c'est bon
        #     if np.all(output == output[0]) == False and i ==0:  ##
        #         previous_output = output  ##

            if isinstance(output, np.ndarray) and len(output)>0:
        #         if np.all(output == output[0]) == False and i >0: #Testing###
        #             if np.allclose(previous_output[1:], output[:-1]) == True:
        #                 previous_output = output
        #             elif np.mean(previous_output[1:] - output[:-1]) / np.median(previous_output[1:]) * 100 < 2.5:
        #                 previous_output = output
        #                 # print("Les résultats entre les fenêtres ne sont pas strictement égaux, mais suffisamment proches")
        #             else:
        #                 print(np.mean(previous_output[1:] - output[:-1]) / np.median(previous_output[1:]) * 100)
        #                 raise Exception(f"{args.AD_Name} does not have constant score on sliding window of size {slidingWindow} and scores have more than 1% difference between successive windows at window {i}")
                        

                final_output[slidingWindow -1 + i]= output[-1]  
            else:
                logging.error(f'At {filename}, sliding window number {i}: '+output)
                errors+=1

            final_run_time[i+1] = run_time

        # Padding scores
        final_output[: slidingWindow-1] = final_output[slidingWindow] 
        normalized_output = normalize_score(final_output, train_index)

        if errors ==0:
            logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {final_run_time.sum():.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', final_output)

        ### whether to save the evaluation result
        if args.save:
            try:
                evaluation_result = get_metrics(final_output, label, slidingWindow=slidingWindow) #slidingWindow for VUS, of no use for us
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
            
            os.makedirs(args.save_dir, exist_ok=True)
            w_csv = pd.DataFrame(write_csv, columns=col_w)            
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)

            np.save(os.path.join(target_time_dir, filename.split('.')[0] + '.npy'), final_run_time)

            normalized_target_dir = target_dir.replace('online', 'online_normalized')
            os.makedirs(normalized_target_dir, exist_ok = True)
            np.save(normalized_target_dir+'/'+filename.split('.')[0]+'.npy', normalized_output)


            