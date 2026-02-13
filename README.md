# StrAD



*In a Streaming World, Should You Stand Still?* A Comprehensive Streaming Anomaly Detection Benchmark

**StrAD** is a large-scale experimental framework designed to evaluate and compare *streaming* versus *static* Time Series Anomaly Detection (TSAD) methods under unified, streaming conditions on reald-world data.



## 📌 Overview

Time series anomaly detection (TSAD) is increasingly deployed instreaming settings, where data arrive sequentially and may exhibitnon-stationarity. As a result, several works from the recent liter-ature propose streaming anomaly detection methods that rely onincremental updates to adapt over time. However, most of theseapproaches originate from the streaming outlier detection litera-ture and largely ignore core characteristics of time series anomalies.Moreover, their empirical evaluation is typically conducted on syn-thetic or small-scale benchmarks with limited diversity, making itunclear whether streaming methods are truly advantageous in real-istic TSAD scenarios. In this work, we carry out the first large-scaleexperimental study comparing streaming and static TSAD methodsunder a unified streaming evaluation benchmark. We consider arealistic setting in which an initial batch of data is available formodel training, followed by online evaluation of both detectionaccuracy and computational efficiency. In addition, we propose adistribution-drift dataset of real time series, called TSB-drift, toisolate scenarios where streaming updates are theoretically jus-tified. Our results show that, contrary to common assumptions,static TSAD methods significantly outperform streaming ap-proaches in most streaming settings. Such finding highlightsa critical gap between the design of existing streaming methodsand the requirements of modern TSAD, and calls for a rethinkingof how streaming capabilities should be integrated into TSAD.

### Key Contributions
* **Large-Scale Study:** Evaluation of **29 TSAD methods** (19 static/online, 10 streaming).
* **Real-World Diversity:** Experiments conducted across **17 diverse real-world datasets**.
* **TSB-drift:** A novel, curated collection of 75 real-world time series exhibiting statistically significant **concept drift**.
* **Unified Evaluation:** Comparison of detection accuracy and computational efficiency in a realistic "batch-train, online-test" pipeline.


## 📊 TSB-drift

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



## 🛠 Benchmark Composition

| Component | Count | Description |
| :--- | :--- | :--- |
| **TSAD Methods (Static/Online)** | 19 | Standard models operating in online settings. |
| **Streaming Methods** | 10 | Methods relying on incremental updates. |
| **Real-World Datasets** | 17 | Diverse multivariate sources (from TSB-AD-M). |
| **TSB-drift Series** | 75 | High-confidence series with identified drift. |



## 📈 Summary of Findings

* **The Temporal Advantage:** Contrary to popular belief, TSAD methods significantly outperform native streaming approaches in most streaming settings.
* **Architectural Limitations:** Streaming methods suffer from a focus on point outliers and their efficient but simple architectures that struggle with subtle temporal dependencies.



## 📂 Project Structure
* `/TSB-drift`: Scripts used to create TSB-drift.
* `/models`: Implementations of the evaluated models.
* `/exp`: Unified streaming evaluation pipeline.
* `/results`: Raw output and visualization scripts for the benchmark results.



## 🚀 Get Started
This project supports three distinct execution modes:

1. Static Execution (Batch Offline)
In this mode, the model (here, TimesNet) is trained on a designated training split and performs a single inference pass over the entire dataset. This is the standard approach for benchmarking.

```python
from models.online.TimesNet import TimesNet
import pandas as pd
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from sklearn import metrics

# Load your data
file_name = '009_MSL_id_8_Sensor_tr_714_1st_1390.csv'
data = pd.read_csv(f"path/to/datasets/{file_name}").iloc[:, 0:-1].values
label = df['Label'].astype(int).to_numpy()
data = df.iloc[:, 0:-1].values.astype(float)

# Run static inference
def run_static_TimesNet(file_name, data):
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict['TimesNet']

    train_index = file_name.split('.')[0].split('_')[-3]
    train_index = int(train_index)
    data_train = data[:train_index, :]
       
    clf = TimesNet(win_size = Optimal_Det_HP['win_size'], lr = Optimal_Det_HP['lr'], enc_in=data_train.shape[1], epochs = 50)
    clf.fit(data_train)
    score = clf.decision_function(data)

    return score

static_score = run_static_TimesNet(file_name, data)

static_AUC_PR = metrics.average_precision_score(label, static_score)

print("Static  (TimesNets) performance AUC-PR: ", static_AUC_PR)
```


2. Online Execution (Sliding Window)
This mode simulates an online environment. The model is pre-trained, but inference is performed window-by-window using a Window3D generator. The anomaly score is updated as each new window "arrives."

```python
from models.online.TimesNet import TimesNet
import pandas as pd
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict
from sklearn import metrics
from models.online.feature import Window3D

# Data is processed as a stream of windows
def run_online_TimesNet(file_name, data):
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict['TimesNet']
    
    train_index = file_name.split('.')[0].split('_')[-3]
    train_index = int(train_index)
    data_train = data[:train_index, :]
    slidingWindow = Optimal_Det_HP['win_size']

    clf = TimesNet(**Optimal_Det_HP, enc_in=data_train.shape[1], epochs = 50)
    clf.fit(data_train)

    data_stream = Window3D(window = slidingWindow).convert(data) 
    score = np.zeros(len(data))
    for i, data_window in enumerate(data_stream):
            output = clf.decision_function(data_window)
            score[slidingWindow -1 + i]= output[-1]  
    score[: slidingWindow-1] = score[slidingWindow] #padding

    return score

online_score = run_online_TimesNet(file_name, data)

online_AUC_PR = metrics.average_precision_score(label, online_score)

print("Online (TimesNets) performance AUC-PR: ", online_AUC_PR)
```



3. Streaming Execution (Real-Time Update)
Designed for true streaming algorithms like MemStream. The model can update its internal memory or state as it processes the stream.

```python
import pandas as pd 
from HP_list import Optimal_Stream_algo_HP_dict
from models.online.feature import Window3D
from models.streaming.MemStream import MemStream
from sklearn import metrics

# Inference performed point-by-point or in unit windows
def run_streaming_MemStream(file_name, data):
    Optimal_Det_HP = Optimal_Stream_algo_HP_dict['MemStream']
    
    train_index = file_name.split('.')[0].split('_')[-3]
    train_index = int(train_index)
    data_train = data[:train_index, :]
    slidingWindow = 1

    clf = MemStream(in_dim=data_train.shape[1], **Optimal_Det_HP)
    clf.fit(data_train)

    data_stream = Window3D(window = slidingWindow).convert(data) 
    score = np.zeros(len(data))
    for i, data_window in enumerate(data_stream):
            output = clf.decision_function(data_window)
            score[i]= output

    return score

streaming_score = run_streaming_MemStream(file_name, data)

streaming_AUC_PR = metrics.average_precision_score(label, streaming_score)
print("Streaming (MemStream) performance AUC-PR: ", streaming_AUC_PR)
```