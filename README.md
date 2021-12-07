# Stress Detection Through Out-Of-Distribution Techniques

Stress is an essential issue of modern society, but the problem lies not only in its presence, but also in the detection and prevention of its negative consequences. Researchers are aware that stress negatively impacts human health and society. Therefore, substantial efforts have lately been made to establish an autonomous stress monitoring system along with various algorithms and methodologies using smart devices. 

In recent years, many works have attempted to detect stress using various machine learning and deep learning algorithms through the classification approach. Even though classification methods perform well for a wide range of datasets, the prior knowledge about the number of classes is required. Since the nature of the stress may differ, it might not be possible to put a finite number of classes for detection. Thus, classification is not the most suitable approach to identify stress. Real life case scenarios may not have independent and identically distributed data samples. 

Due to the fact that stress data is rare and has a unique statistical distribution from normal data, an alternative method to use is Out-Of-Distribution (OOD) techniques, which are widely used for the  detection of anomalies. These methods are needed for identifying new samples that might not belong to the in-distribution class. For instance, if cats are the in-distribution data then dogs, horses and bears are considered as out of distribution data \cite{boyer2021out}. Similarly, in a stress detection problem the normal behaviors such as reading, and meditation are identified as in-distribution while stress causing activities like exams and workouts are classified as out-of-distribution. Therefore, stress data samples in a dataset can be treated as outliers.

In this project, we aim to investigate the possibility of performing stress detection by applying out-of-distribution (OOD) concept through anomaly detection instead of using usual classification methods. Although OOD techniques have long existed for image classification but it is less widely applied in the context of time series stress detection data. We will approach our objective through utilizing a well-known dataset in the field of stress detection WESAD \cite{schmidt2018introducing} and applying three different machine learning algorithms on it. The goal is to compare the performances of our chosen algorithms with that of existing work and analyze how well they perform on this dataset with the hypothesis of using OOD techniques.


Our best performing result is also compared to the SOTA deep learning hierarchical model \cite{kumar2021hierarchical}. In order to compare using the same metrics used by \cite{kumar2021hierarchical}, we also calculated the F1-score which has a formula of $2 \times \frac{Precision \times Recall}{Precision + Recall}$. The isolation forest algorithm using cleaned data together with removed overlapping data points is able to achieve 91.47\% accuracy, outperforming the SOTA model, which only has an accuracy of 87.7\% that corresponds to the average of classification accuracy among 15 subjects. The F1-score for the isolation forest is also much better at 0.916 while SOTA is only at 0.83. 

Stress has been a serious mental issue in the modern society. It can only be detected by a trained medical professional. However, early detection of stress signals can be a lifesaver in some situations. Therefore, an automatic stress detection algorithm is helpful in such cases when medical professional are not available for the patient. Stress does not occur as often as compared to the rest of the human emotions, so gathering stress data can be a difficult task. This is where our proposed out-of-distribution detection method for stress detection comes in handy. But, in order to use OOD detection methods, the dataset is very important no matter what kind of model it is applied on. The following assumptions has to be satisfied for the dataset, which is the in-distribution data has to be available in abundance and the out-of-distribution data is drawn from a very different distribution than the training set and is only available in small quantity. When compared to the SOTA deep learning model which is based on classification, our project proved that using OOD technique in such scenario with carefully pre-processed data can achieve better results and higher accuracy especially in the case of detecting one emotion.
