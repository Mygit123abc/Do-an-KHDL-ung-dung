# Generative AI and Deepfake Detection

This repository contains the code and data for the paper published at ICLR 2021, which explores the current state of generative AI models and forensic techniques for deepfake detection.

## Table of Contents
1. [Introduction](#introduction)
2. [Generative Models](#generative-models)
    - [Unconditional Generative Model](#unconditional-generative-model)
    - [Text-to-Image Generative Model](#text-to-image-generative-model)
3. [Deepfake Detection](#deepfake-detection)
    - [Deepfake Classification](#deepfake-classification)
    - [Video Face Manipulation Detection](#video-face-manipulation-detection)
4. [Datasets](#datasets)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
With deepfake technology being rapidly improved and new generative techniques being developed, it is important to understand the current state of generative AI as well as its counter-measures through forensic techniques.

## Generative Models

### Unconditional Generative Model
We experimented with different hyper-parameters on Diffusion models, achieving varied image quality and inference times. The results showed that Diffusion models, with high time steps, fail to achieve high performance due to dataset imbalance. Further experiments with the white-noise method proposed by Kong & Ping (2021) showed image quality degradation in the initial epochs but failed to produce meaningful results with higher steps. DCGAN also struggled with 64x64 resolution images, capturing some facial features but failing to produce smooth pixels.

### Text-to-Image Generative Model
DALL-E reported zero-shot results on MS-COCO, achieving better FID and IS scores than other state-of-the-art approaches. However, we have not yet conducted experiments in this area.

## Deepfake Detection

### Deepfake Classification
There has been no prior work on the Digiface dataset for deepfake classification. This should be explored in future research.

### Video Face Manipulation Detection
Experiments with four EfficientNet-based models showed similar results to the original experiment by Bonettini et al. (2020), with EfficientNetB4 achieving the best performance. This indicates that each model can work well independently without needing to combine multiple networks.

## Datasets

### LFW Dataset
The LFW dataset contains 5749 identities with a total of 13,233 face images. The dataset has a severe imbalance problem, which could potentially harm the performance of Diffusion models.

### COCO-Captions
COCO-Captions is a large-scale dataset containing pairs of images and captions. We applied tf-idf ranking algorithms on the captions to calculate their relevance to associated image labels, finding some annotation errors and imbalances.

### Digiface-1M
The Digiface-1M dataset from Microsoft contains varying poses, genders, ages, illumination, and saturation. We conducted several analyses on these features using pre-trained models and found that light conditions do not significantly affect facial features.

### DFDC
The DFDC dataset, used for the Deepfake Detection Challenge, contains over 120,000 video sequences representing both real and artificially made videos. It is used to train forensic techniques for detecting manipulated faces in video.

## Conclusion
The goal of this paper is to display the capabilities of both generating and detecting technology, as well as presenting the progress of the "arms race" so far. In the future, we hope to further research state-of-the-art generative AI models to gain a better understanding of the field and improve forensic techniques to combat the growing threat of deepfakes and generative AI abuse.

## References
- Gwangbin Bae, Martin de La Gorce, Tadas Baltrusaitis, Charlie Hewitt, Dong Chen, Julien Valentin, Roberto Cipolla, and Jingjing Shen. Digiface-1m: 1 million digital face images for face recognition, 2022.
- Nicolò Bonettini, Edoardo Daniele Cannas, Sara Mandelli, Luca Bondi, Paolo Bestagini, and Stefano Tubaro. Video face manipulation detection through ensemble of cnns, 2020.
- Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server, 2015.
- François Chollet. Xception: Deep learning with depthwise separable convolutions, 2017.
- Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks, 2019.
- Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models, 2021.
- Neeraj Kumar, Alexander Berg, Peter N. Belhumeur, and Shree Nayar. Describable visual attributes for face verification and image search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(10):1962–1977, 2011. doi: 10.1109/TPAMI.2011.48.
- Maksim Kuprashevich and Irina Tolstykh. Mivolo: Multi-input transformer for age and gender estimation. 2023.
- Maksim Kuprashevich, Grigorii Alekseenko, and Irina Tolstykh. Beyond specialization: Assessing the capabilities of mllms in age and gender estimation. 2024.
- Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets, 2014.
- Yiming Qin, Huangjie Zheng, Jiangchao Yao, Mingyuan Zhou, and Ya Zhang. Class-balancing diffusion models, 2023.
- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks, 2020.
