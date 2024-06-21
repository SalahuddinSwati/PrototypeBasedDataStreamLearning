# PrototypeBasedDataStreamLearning
Title: A reliable adaptive prototype-based learning for evolving data streams with limited labels
-------------------------------------------------------------------------------------------------------------------------------
Abstract
-------------------------------------------------------------------------------------------------------------------------------
Data stream mining presents notable challenges in the form of concept drift and evolution. Existing learning algorithms, typically designed within a supervised learning framework, require class labels for all data points. However, this is an impractical requirement given the rapid pace of data streams, which often results in label scarcity. Recognizing the realistic necessity of learning from data streams with limited labels, we propose an adaptive, data-driven, prototype-based semi-supervised learning framework specifically tailored to handle evolving data streams. Our method employs a prototype-based data representation, summarizing the continuous flow of streaming data using dynamic prototypes at varying levels of granularity. This technique enables improved data abstraction, capturing the underlying local data distributions more accurately. The model also incorporates reliability modeling and efficient emerging class discovery, dynamically updating the significance of prototypes over time and swiftly adapting to local concept drift. We further leverage these adaptive prototypes to intuitively detect concept evolution, i.e., identifying novel classes from a local density perspective. To minimize the need for manual labeling while optimizing performance, we incorporate active learning into our method. This method employs a dual-criteria approach for data point selection, considering both uncertainty and local density. These manually labeled data points, together with unlabeled data, serve to update the model efficiently and robustly. Empirical validation using several benchmark datasets demonstrates promising performance in comparison to existing state-of-the-art techniques.

-------------------------------------------------------------------------------------------------------------------------------

This is the version 1, and it will be constantly improved. We will update the progress.

-------------------------------------------------------------------------------------------------------------------------------

Reference: Din, Salah Ud, Aman Ullah, Cobbinah B. Mawuli, Qinli Yang, and Junming Shao. "A reliable adaptive prototype-based learning for evolving data streams with limited labels." Information Processing & Management 61, no. 1 (2024): 103532.

-------------------------------------------------------------------------------------------------------------------------------
ATTN: This code were developed by Salah Ud Din (salahuddin@csj.uestc.edu.cn). For any problem and suggestment, please feel free to contact Dr. Salah Ud Din.
