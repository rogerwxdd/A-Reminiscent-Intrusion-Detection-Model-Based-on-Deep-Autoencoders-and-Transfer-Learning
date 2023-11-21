# A Reminiscent Intrusion Detection Model Based on Deep Autoencoders and Transfer Learning

## Authors: Roger R. dos Santos and Eduardo K. Viegas and Altair O. Santin


 Machine learning techniques for network-based intrusion detection often assumes that network traffic does not change over time, or that model updates can be easily performed. In this paper we propose a novel reminiscent intrusion detection model based on deep autoencoders and transfer learning to easiness the model update burden, implemented twofold. First, deep autoencoder is used as an additional feature extraction stage to extract an historical feature representation of network traffic. Second, at model updates, the deep autoencoder parameters are updated through a transfer learning procedure, thus, significantly decreasing the amount of needed labeled training data and the computational costs. Experiments performed on dataset with 8TB of data, containing real and valid network traffic that span for a year, have shown that approaches in the literature are unable to deal with network traffic changes over time, while also demanding unfeasible amounts of labeled data during model training. In addition, if no model updates are performed, the proposed scheme is able to improve the true-negative rate by up to 23.9%, if done so, it is able to provide similar accuracy rates of traditional techniques while demanding only 22% of labeled training data and 28% of computational costs.
 

## Proposal
# ![image](https://github.com/rogerwxdd/A-Reminiscent-Intrusion-Detection-Model-Based-on-Deep-Autoencoders-and-Transfer-Learning/assets/151274633/b9d742cf-54fb-4201-82d7-edcd2f06bd85)

## Dataset
#

## BibTex

```.bib
@INPROCEEDINGS{9685724,
  author={dos Santos, Roger R. and Viegas, Eduardo K. and Santin, Altair O.},
  booktitle={2021 IEEE Global Communications Conference (GLOBECOM)}, 
  title={A Reminiscent Intrusion Detection Model Based on Deep Autoencoders and Transfer Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/GLOBECOM46510.2021.9685724}}
```
