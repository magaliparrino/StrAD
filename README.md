# StrAD Benchmark



*In a Streaming World, Should You Stand Still?* A Comprehensive Streaming Anomaly Detection Benchmark

**StrAD** is a large-scale experimental framework designed to evaluate and compare *streaming* versus *static* Time Series Anomaly Detection (TSAD) methods under unified, streaming conditions on real-world data.



##  Overview

Time series anomaly detection (TSAD) is increasingly deployed in streaming settings, where data arrive sequentially and may exhibit non-stationarity. As a result, several works from the recent literature propose streaming anomaly detection methods that rely on incremental updates to adapt over time. However, most of these approaches originate from the streaming outlier detection literature and largely ignore core characteristics of time series anomalies. Moreover, their empirical evaluation is typically conducted on synthetic or small-scale benchmarks with limited diversity, making it unclear whether streaming methods are truly advantageous in realistic TSAD scenarios. In this work, we carry out the first large-scale experimental study comparing streaming and static TSAD methods under a unified streaming evaluation benchmark. We consider a realistic setting in which an initial batch of data is available for model training, followed by online evaluation of both detection accuracy and computational efficiency. In addition, we propose a distribution-drift dataset of real time series, called TSB-drift, to isolate scenarios where streaming updates are theoretically justified. Our results show that, contrary to common assumptions, static TSAD methods significantly outperform streaming approaches in most streaming settings. Such finding highlights a critical gap between the design of existing streaming methods and the requirements of modern TSAD, and calls for a rethinking of how streaming capabilities should be integrated into TSAD.

### Key Contributions
* **Large-Scale Study:** Evaluation of **29 TSAD methods** (19 static/online, 10 streaming).
* **Real-World Diversity:** Experiments conducted across **17 diverse real-world datasets**.
* **TSB-drift:** A curated collection of 75 real-world time series exhibiting statistically significant **concept drift**.
* **Unified Evaluation:** Comparison of detection accuracy and computational efficiency in a realistic "batch-train, online-test" pipeline.



## Installation

Installation requires `git` and `conda`

```bash
git clone https://github.com/magaliparrino/StrAD.git
cd StrAD
conda create -n StrAD python=3.11
conda activate StrAD
pip install -r requirements.txt
```

##  Get Started
This project supports three distinct execution modes:

#### 1. Static Execution (Batch Offline)
In this mode, the model (here, TimesNet) is trained on a designated training split and performs a single inference pass over the entire dataset. This is the standard approach for benchmarking, and the one used in TSB-AD.

```python
import pandas as pd
from exp.online_model_wrapper import fit_AD
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from sklearn import metrics

# Load your data
file_name = '009_MSL_id_8_Sensor_tr_714_1st_1390.csv'
df = pd.read_csv(f"path/to/datasets/{file_name}")
label = df['Label'].astype(int).to_numpy()
data = df.iloc[:, 0:-1].values.astype(float)
train_index = file_name.split('.')[0].split('_')[-3]
train_index = int(train_index)
data_train = data[:train_index, :]

model_name = 'TimesNet' # or any other model of the static pool

Optimal_Det_HP = Optimal_Multi_algo_HP_dict[model_name]

# Run the model
clf = fit_AD(model_name,data_train, **Optimal_Det_HP) 
static_score = clf.decision_function(data)

static_AUC_PR = metrics.average_precision_score(label, static_score)

print("Static (TimesNets) performance AUC-PR: ", static_AUC_PR)
```


#### 2. Online Execution (Sliding Window)
This mode simulates an online environment. The model is pre-trained on historical data and the inference is performed window-by-window using a Window3D generator. The anomaly score is updated as each new window "arrives."

```python
import pandas as pd
import numpy as np
from exp.online_model_wrapper import fit_AD
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from sklearn import metrics
from models.online.feature import Window3D

# Load your data
file_name = '009_MSL_id_8_Sensor_tr_714_1st_1390.csv'
df = pd.read_csv(f"path/to/datasets/{file_name}")
label = df['Label'].astype(int).to_numpy()
data = df.iloc[:, 0:-1].values.astype(float)
train_index = file_name.split('.')[0].split('_')[-3]
train_index = int(train_index)
data_train = data[:train_index, :]

model_name = 'TimesNet' # or any other model of the online pool
Optimal_Det_HP = Optimal_Multi_algo_HP_dict[model_name]
slidingWindow = Optimal_Det_HP['win_size']

clf = fit_AD(model_name,data_train, **Optimal_Det_HP)

# Data is processed as a stream of windows
data_stream = Window3D(window = slidingWindow).convert(data) 
online_score = np.zeros(len(data))
for i, data_window in enumerate(data_stream):
        output = clf.decision_function(data_window)
        online_score[slidingWindow -1 + i]= output[-1]  
online_score[: slidingWindow-1] = online_score[slidingWindow] #padding

online_AUC_PR = metrics.average_precision_score(label, online_score)

print("Online (TimesNets) performance AUC-PR: ", online_AUC_PR)
```



#### 3. Streaming Execution (Real-Time Update)
Designed for true streaming algorithms like MemStream. The model can update its internal memory or state as it processes the stream.

```python
import pandas as pd 
import numpy as np
from exp.streaming_model_wrapper import fit_AD
from HP_list import Optimal_Stream_algo_HP_dict
from models.online.feature import Window3D
from sklearn import metrics

# Load your data
file_name = '009_MSL_id_8_Sensor_tr_714_1st_1390.csv'
df = pd.read_csv(f"path/to/datasets/{file_name}")
label = df['Label'].astype(int).to_numpy()
data = df.iloc[:, 0:-1].values.astype(float)
train_index = file_name.split('.')[0].split('_')[-3]
train_index = int(train_index)
data_train = data[:train_index, :]
slidingWindow = 1

model_name = "MemStream" # or any other model in the streaming pool
Optimal_Det_HP = Optimal_Stream_algo_HP_dict[model_name]

clf = fit_AD(model_name,data_train, **Optimal_Det_HP)

# Inference performed point-by-point or in unit windows
data_stream = Window3D(window = slidingWindow).convert(data) 
streaming_score = np.zeros(len(data))
for i, data_window in enumerate(data_stream):
        output = clf.decision_function(data_window)
        streaming_score[i]= output

streaming_AUC_PR = metrics.average_precision_score(label, streaming_score)
print("Streaming (MemStream) performance AUC-PR: ", streaming_AUC_PR)
```


##  TSB-drift

To isolate scenarios where streaming updates are theoretically justified, we introduce **TSB-drift**. While most streaming anomaly detection benchmarks rely on synthetic drift, TSB-drift identifies distributional changes in real-world data using a systematic four-step process:

1. **Batch Subdivision:** Series are divided into batches of size $t_r$ (the training size).
2. **Distributional Change Measurement:** We quantify the **Jensen-Shannon Divergence ($JSD$)** between batch pairs $(B_i, B_k)$.
   $$JSD(B_i \parallel B_k) = \frac{1}{2} D_{KL}(B_i \parallel A_{ik}) + \frac{1}{2} D_{KL}(B_k \parallel A_{ik})$$
   where $A_{ik} = \frac{1}{2}(B_i + B_k)$.
3. **Max-Pooling Aggregation:** We capture drift occurring in even a single dimension by taking the maximum divergence across all dimensions: $M_{i,k} = \max_j J_{ik}^{(j)}$.
4. **Ranking & Selection:** We selected the top 75 series exhibiting the most pronounced and long-lasting drifts.

### Identified Drift Patterns
| Pattern | Type | Visual Characteristics |
| :--- | :--- | :--- |
| **Continuous (C)** | Gradual | Diagonal heatmaps |
| **Change Points (CP)** | Abrupt | Block-matrix structures |
| **Periodic (P)** | Reoccurring | Cobbled/Grid-like heatmaps |
| **Random Walks (RW)** | Unstructured | No distinctive patterns |



## Benchmark Composition

| Component | Count | Description |
| :--- | :--- | :--- |
| **TSAD Methods (Static/Online)** | 19 | Standard models operating in online settings. |
| **Streaming Methods** | 10 | Methods relying on incremental updates. |
| **Real-World Datasets** | 17 | Diverse multivariate sources (from TSB-AD-M). |
| **TSB-drift Series** | 75 | High-confidence series with identified drift. |



## Summary of Findings

* **Architectural Limitations:** Streaming methods suffer from a focus on point outliers and their efficient but simple architectures that struggle with subtle temporal dependencies.
* **The Temporal Advantage:** Contrary to popular belief, TSAD methods significantly outperform native streaming approaches in most streaming settings.


## Project Structure
* `/TSB-drift`: Scripts used to create TSB-drift.
* `/models`: Implementations of the evaluated models.
* `/exp`: Unified streaming evaluation pipeline.
* `/results`: Raw output and visualization scripts for the benchmark results.

## Citation
If you use this work, please consider citing the associated paper
```bibtex
@inproceedings{parrino:hal-05654228,
  TITLE = {{In a Streaming World, Should You Stand Still? A Comprehensive Benchmark of Anomaly Detection in Streams}},
  AUTHOR = {Parrino, Magali and Ajenjo, Antoine and Remy, Emmanuel and Stephan, Pierre and Senellart, Pierre and Boniol, Paul},
  URL = {https://inria.hal.science/hal-05654228},
  BOOKTITLE = {{Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)}},
  ADDRESS = {Jeju, South Korea},
  YEAR = {2026},
  MONTH = Aug,
  DOI = {10.1145/3770855.3817495},
  KEYWORDS = {Data stream mining ; Time series analysis ; Time Series ; Benchmark ; Anomaly Detection ; Stream},
  PDF = {https://inria.hal.science/hal-05654228v1/file/v2dtb173_CameraReady.pdf},
  HAL_ID = {hal-05654228},
  HAL_VERSION = {v1},
}
```

## Acknowledgements

This repository is built upon the open-source implementation of **TSB-AD** developed by TheDatumOrg. We sincerely thank the authors for providing their benchmarking framework, which served as the foundation for our experiments.

* https://github.com/TheDatumOrg/TSB-AD

Many thanks to the following repos for their invaluable code base:
* https://github.com/CN-TU/dSalmon/tree/master
* https://infolab.usc.edu/Luan/Outlier/CountBasedWindow/DODDS/src/outlierdetection/
* https://pysad.readthedocs.io/en/latest/