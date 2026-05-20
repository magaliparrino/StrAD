# Time Series Anomaly Detection (TSAD) Methods in StrAD

This document contains the tables for Static/Online and Streaming Time Series Anomaly Detection (TSAD) methods evaluated in **StrAD**, the bibliographic references integrated.

---

### Table 1: Static/Online TSAD Methods in **StrAD**
*Note: Implementations come from the TSB-AD benchmark [[29](#ref-tsbad)].*

| Acronym | Method | Type |
| :--- | :--- | :--- |
|  | **Distance-based** | |
| **LOF** | LOF [[1](#ref-lof)] | Proximity |
| **KNN** | $k$-NN [[2](#ref-hawkins80)] | Proximity |
| **KMAD** | $k$-Means [[2](#ref-hawkins80)] | Clustering |
| **CBLOF** | CBLOF [[3](#ref-cblof)] | Clustering |
| | **Density-based** | |
| **IF** | Isolation Forest [[4](#ref-iforest)] | Tree |
| **MCD** | MCD [[5](#ref-mcd)] | Distribution |
| **HBOS** | HBOS [[6](#ref-hbos)] | Distribution |
| **SVM** | OCSVM [[7](#ref-ocsvm)] | Distribution |
| **PCA** | PCA [[8](#ref-pca)] | Encoding |
| **RPCA** | RobustPCA [[9](#ref-robustpca)] | Encoding |
| |  **Prediction-based**| |
| **CNN** | CNN [[10](#ref-cnn)] | Forecasting |
| **LSTM** | LSTMAD [[11](#ref-lstmad)] | Forecasting |
| **AT** | AnomalyTransformer [[12](#ref-anomalytransformer)] | Reconstruction |
| **AE** | AutoEncoder [[13](#ref-autoencoder)] | Reconstruction |
| **TrAD** | TranAD [[14](#ref-tranad)] | Reconstruction |
| **TN** | TimesNet [[15](#ref-timesnet)] | Reconstruction |
| **USAD** | USAD [[16](#ref-usad)] | Reconstruction |
| **OA** | OmniAnomaly [[17](#ref-omnianomaly)] | Reconstruction |
| **FITS** | FITS [[18](#ref-fits)] | Reconstruction |

---

### Table 2: Streaming TSAD in **StrAD**

| Acronym | Method | UM (Sec 3.1) | MM (Sec 3.2) |
| :--- | :--- | :--- | :--- |
| | |**Numerical**  | |
| **LODA** | LODA [[19](#ref-loda)] | Projections | Tumbling Window |
| **xS** | xStream [[20](#ref-xstream)] | Projections | Tumbling Window |
| **RSH** | RSHash [[21](#ref-rshash)] | Partitioning | Sliding Window (Point) |
| **HST** | HSTree [[22](#ref-hstree)] | Partitioning | Tumbling Window |
| **SDOs** | SDOstream [[23](#ref-sdostream)] | Partitioning | Soft Forgetting (Aging) |
|  | | **Structural** | |
| **RRCF** | RRCF [[24](#ref-rrcf)] | Tree | Sliding Window (Point) |
| **MCOD** | MCOD [[25](#ref-mcod)] | Clustering | Sliding Window (Point) |
| **LEAP** | LEAP [[26](#ref-leap)] | Proximity | Sliding Window (Batch) |
| **SKNN** | SWKNN [[27](#ref-swknn)] | Proximity | Sliding Window (Point) |
| **MemS** | MemStream [[28](#ref-memstream)] | Encoding | Soft Forgetting (Selective) |

---

## References

<a id="ref-lof"></a>**[1]** M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander. "LOF: Identifying Density-Based Local Outliers." *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 2000, pp. 93–104. [DOI](https://doi.org/10.1145/342009.335388)

<a id="ref-hawkins80"></a>**[2]** D. M. Hawkins. *Identification of Outliers*. Springer, 1980. [DOI](https://doi.org/10.1007/978-94-015-3994-4)

<a id="ref-cblof"></a>**[3]** Z. He, X. Xu, and S. Deng. "Discovering cluster-based local outliers." *Pattern Recognition Letters*, vol. 24, no. 9-10, 2003, pp. 1641–1650. [DOI](https://doi.org/10.1016/S0167-8655(03)00003-5)

<a id="ref-iforest"></a>**[4]** F. T. Liu, K. M. Ting, and Z.-H. Zhou. "Isolation Forest." *Proceedings of the 8th IEEE International Conference on Data Mining (ICDM 2008)*, 2008, pp. 413–422. [DOI](https://doi.org/10.1109/ICDM.2008.17)

<a id="ref-mcd"></a>**[5]** P. J. Rousseeuw. "Least Median of Squares Regression." *Journal of the American Statistical Association*, vol. 79, no. 388, 1984, pp. 871–880. [DOI](https://doi.org/10.1080/01621459.1984.10477105)

<a id="ref-hbos"></a>**[6]** M. Goldstein and A. R. Dengel. "Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm." *KI*, 2012.

<a id="ref-ocsvm"></a>**[7]** B. Schölkopf, R. C. Williamson, A. J. Smola, J. Shawe-Taylor, and J. C. Platt. "Support Vector Method for Novelty Detection." *Advances in Neural Information Processing Systems 12 (NIPS 1999)*, 1999, pp. 582–588.

<a id="ref-pca"></a>**[8]** M.-L. Shyu, S.-C. Chen, K. Sarinnapakorn, and L. Chang. "A Novel Anomaly Detection Scheme Based on Principal Component Classifier." *Foundations and New Directions of Data Mining Workshop at ICDM 2003*, 2003.

<a id="ref-robustpca"></a>**[9]** R. C. Paffenroth, K. Kay, and L. Servi. "Robust PCA for Anomaly Detection in Cyber Networks." *CoRR*, abs/1801.01571, 2018.

<a id="ref-cnn"></a>**[10]** M. Munir, S. A. Siddiqui, A. Dengel, and S. Ahmed. "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series." *IEEE Access*, vol. 7, 2019, pp. 1991–2005. [DOI](https://doi.org/10.1109/ACCESS.2018.2886457)

<a id="ref-lstmad"></a>**[11]** P. Malhotra, L. Vig, G. Shroff, and P. Agarwal. "Long Short Term Memory Networks for Anomaly Detection in Time Series." *23rd European Symposium on Artificial Neural Networks (ESANN 2015)*, 2015.

<a id="ref-anomalytransformer"></a>**[12]** J. Xu, H. Wu, J. Wang, and M. Long. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." *The Tenth International Conference on Learning Representations (ICLR 2022)*, 2022.

<a id="ref-autoencoder"></a>**[13]** M. Sakurada and T. Yairi. "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction." *Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis*, 2014, p. 4. [DOI](https://doi.org/10.1145/2689746.2689747)

<a id="ref-tranad"></a>**[14]** S. Tuli, G. Casale, and N. R. Jennings. "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *Proc. VLDB Endow.*, vol. 15, no. 6, 2022, pp. 1201–1214. [DOI](https://doi.org/10.14778/3514061.3514067)

<a id="ref-timesnet"></a>**[15]** H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." *The Eleventh International Conference on Learning Representations (ICLR 2023)*, 2023.

<a id="ref-usad"></a>**[16]** J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga. "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *KDD '20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2020, pp. 3395–3404. [DOI](https://doi.org/10.1145/3394486.3403392)

<a id="ref-omnianomaly"></a>**[17]** Y. Su, Youjian Zhao, C. Niu, R. Liu, W. Sun, and D. Pei. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2019, pp. 2828–2837. [DOI](https://doi.org/10.1145/3292500.3330672)

<a id="ref-fits"></a>**[18]** Z. Xu, A. Zeng, and Q. Xu. "FITS: Modeling Time Series with 10k Parameters." *The Twelfth International Conference on Learning Representations (ICLR 2024)*, 2024.

<a id="ref-loda"></a>**[19]** T. Pevný. "Loda: Lightweight on-line detector of anomalies." *Machine Learning*, vol. 102, no. 2, 2016, pp. 275–304. [DOI](https://doi.org/10.1007/s10994-015-5521-0)

<a id="ref-xstream"></a>**[20]** E. A. Manzoor, H. Lamba, and L. Akoglu. "xStream: Outlier Detection in Feature-Evolving Data Streams." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018, pp. 1963–1972. [DOI](https://doi.org/10.1145/3219819.3220107)

<a id="ref-rshash"></a>**[21]** S. Sathe and C. C. Aggarwal. "Subspace histograms for outlier detection in linear time." *Knowledge and Information Systems*, vol. 56, no. 3, 2018, pp. 691–715. [DOI](https://doi.org/10.1007/s10115-017-1148-8)

<a id="ref-hstree"></a>**[22]** S. C. Tan, K. M. Ting, and F. T. Liu. "Fast Anomaly Detection for Streaming Data." *IJCAI 2011, Proceedings of the 22nd International Joint Conference on Artificial Intelligence*, 2011, pp. 1511–1516. [DOI](https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-254)

<a id="ref-sdostream"></a>**[23]** A. Hartl, F. Iglesias, and T. Zseby. "SDOstream: Low-Density Models for Streaming Outlier Detection." *28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2020)*, 2020, pp. 661–666.

<a id="ref-rrcf"></a>**[24]** S. Guha, N. Mishra, G. Roy, and O. Schrijvers. "Robust Random Cut Forest Based Anomaly Detection on Streams." *Proceedings of the 33nd International Conference on Machine Learning (ICML 2016)*, 2016, pp. 2712–2721.

<a id="ref-mcod"></a>**[25]** M. Kontaki, A. Gounaris, A. N. Papadopoulos, K. Tsichlas, and Y. Manolopoulos. "Continuous monitoring of distance-based outliers over data streams." *Proceedings of the 27th International Conference on Data Engineering (ICDE 2011)*, 2011, pp. 135–146. [DOI](https://doi.org/10.1109/ICDE.2011.5767923)

<a id="ref-leap"></a>**[26]** L. Cao, D. Yang, Q. Wang, Y. Yu, J. Wang, and E. A. Rundensteiner. "Scalable distance-based outlier detection over high-volume data streams." *IEEE 30th International Conference on Data Engineering (ICDE 2014)*, 2014, pp. 76–87. [DOI](https://doi.org/10.1109/ICDE.2014.6816641)

<a id="ref-swknn"></a>**[27]** S. Ramaswamy, R. Rastogi, and K. Shim. "Efficient Algorithms for Mining Outliers from Large Data Sets." *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 2000, pp. 427–438. [DOI](https://doi.org/10.1145/342009.335437)

<a id="ref-memstream"></a>**[28]** S. Bhatia, A. Jain, S. Srivastava, K. Kawaguchi, and B. Hooi. "MemStream: Memory-Based Streaming Anomaly Detection." *WWW '22: The ACM Web Conference 2022*, 2022, pp. 610–621. [DOI](https://doi.org/10.1145/3485447.3512221)

<a id="ref-tsbad"></a>**[29]** Q. Liu, and J. Paparrizos. "The Elephant in the Room: Towards {A} Reliable Time-Series Anomaly Detection Benchmark." *WWW '22: The ACM Web Conference 2022*, 2022, pp. 610–621. [DOI](https://doi.org/10.1145/3485447.3512221)