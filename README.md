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
* `/data`: Scripts to download and preprocess TSB-AD-M and TSB-drift.
* `/methods`: Implementations of the 29 evaluated models.
* `/evaluation`: Unified streaming evaluation pipeline.
* `/results`: Raw output and visualization scripts for the benchmark results.

