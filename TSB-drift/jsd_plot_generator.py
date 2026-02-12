import pandas as pd
from jsd_drift import compute_kl_matrices, aggregate_feature_matrices, normalize_df_zscore
from plot_jsd_heatmaps import plot_feature_heatmaps_grid, plot_single_heatmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pickle


if __name__ == '__main__':
    FILE_DIR = 'Datasets/'#'../../../TSBAD/Datasets/TSB-AD-M/'
    file_list = ['064_SMD_id_8_Facility_tr_2272_1st_2372.csv']#pd.read_csv('../../../TSBAD/Datasets/File_List/TSB-AD-M-Eva.csv')['file_name'].values
    output_dir_plot = "./heatmaps/"
    output_dir_mat = "./analysis/"

    for file in file_list:

        df = pd.read_csv(FILE_DIR + file).dropna()

        data = df.iloc[:, 0:-1].values.astype(float)

        train_index = file.split('.')[0].split('_')[-3]
        train_index = int(train_index)

        BATCH_LEN = train_index  
        FEATURES = df.columns.drop('Label')    

        df_norm, scaler = normalize_df_zscore(df)

        mats, batch_labels, edges = compute_kl_matrices(
            df_norm, batch_len=BATCH_LEN, features=FEATURES,
            bins='auto', alpha=1e-2, metric="jsd", drop_incomplete=True # "kl" | "jsd" | "wasserstein"
        )

        mean_mat = aggregate_feature_matrices(mats, how = "mean")
        max_mat = aggregate_feature_matrices(mats, how = "max")

        mean_path = os.path.join(output_dir_mat, 'mean')
        max_path = os.path.join(output_dir_mat, 'max')

        os.makedirs(output_dir_plot, exist_ok=True)
        os.makedirs(os.path.join(output_dir_mat, 'mean'), exist_ok=True)
        os.makedirs(os.path.join(output_dir_mat, 'max'), exist_ok=True)
        os.makedirs(os.path.join(output_dir_mat, 'mats'), exist_ok=True)

        np.save(mean_path+'/'+file.split('.')[0]+'.npy', mean_mat)
        np.save(max_path+'/'+file.split('.')[0]+'.npy', max_mat)

        path = output_dir_mat+'/mats/'+file.split('.')[0]+'.pkl'
        with open(path, 'wb') as f:
            pickle.dump(mats, f)


        sns.set_style("white")
        n_cols = 5

        if data.shape[1] > 60:
            n_cols = 10

        fig, axes = plot_feature_heatmaps_grid(
            mats, batch_labels,
            ncols=n_cols, share_color_scale=True, percentile_clip=99.0,
            suptitle=f"JSD(b_i || b_j) per feature for TS {file}",
            use_jsd_range=True
        )

        
        root, ext = os.path.splitext(file)
        png_name = f"{root}.png"

        out_path = os.path.join(output_dir_plot, png_name)

        
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)  

        print(f"file {file} done")
        

