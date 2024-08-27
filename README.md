# MuSe-2024
Multimodal Humor Detection and Social Perception Prediction

## IAF and Late Fusion:
The code for this experiment is available [here](https://github.com/aalto-speech/MuSe-2024/tree/main/baseline_and_iaf). This code also includes the baseline experiment and the modality contributions analysis (Section 5.1 in our paper).

## Transformer-based Fusion for Humor:
The code for this experiment  is available [here](https://github.com/aalto-speech/MuSe-2024/tree/main/humor_trf). This code reproduces the results of the TrF models in Table 1 of our paper.

## Joint Decoder for Perception:
The code for this experiment  is available [here](https://github.com/aalto-speech/MuSe-2024/tree/main/perception_joint_decoder). This experiment primarily uses the [SpeechBrain](https://github.com/speechbrain/speechbrain) toolkit. You can also modify the experiment to evaluate different segments of the recording, such as the first 10 seconds and last 10 seconds of each sample (Section 5 in our paper).

## Opposing Trait-Based Ensemble:
The trait-based ensemble is implemented in the [JupyterNotebook](https://github.com/aalto-speech/MuSe-2024/blob/main/trait_based_ensemble.ipynb). The trait selection for each group is based on the correlation analysis of the training set (see [Figure](https://github.com/aalto-speech/MuSe-2024/blob/main/perception_correlation.png))

## Citation:

Mehedi Hasan Bijoy, Dejan Porjazovski, Nhan Phan, Guangpu Huang, Tamás Grósz, and Mikko Kurimo. 2024. Multimodal Humor Detection and Social Perception Prediction. In _Proceedings of the 5th Multimodal Sentiment Analysis Challenge and Workshop: Social Perception and Humor (MuSe ’24), October 28-November 1, 2024, Melbourne, VIC, Australia_. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3689062.3689376

```tex
@inproceedings{bijoy2024multimodal,
    author = {Bijoy, Mehedi Hasan and Porjazovski, Dejan and Phan, Nhan and Huang, Guangpu and Grósz, Tamás and Kurimo, Mikko},
    title = {Multimodal Humor Detection and Social Perception Prediction},
    booktitle = {Proceedings of the 5th Multimodal Sentiment Analysis Challenge and Workshop: Social Perception and Humor (MuSe '24)},
    year = {2024},
    address = {Melbourne, VIC, Australia},
    publisher = {ACM, New York, NY, USA},    
    doi = {10.1145/3689062.3689376},
}
