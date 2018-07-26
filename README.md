[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/GilesStrong/QCHS-2018/master)

# Recent developments in deep-learning applied to open HEP data
A lot of work done in advancing the performance of deep-learning approaches often takes place in the realms of image recognition - many papers use famous benchmark datasets, such as Cifar or Imagenet, to quantify the advantages their idea offers. However it is not always obvious, when reading such papers, whether the concepts presented can also be applied to problems in other domains and still offer improvements.

One such example of another domain is the task of event classification in high-energy particle-collisions, such as those which occur at the LHC. In this presentation, a classifier trained on publicly available physics data (from the [HiggsML Kaggle challenge](https://www.kaggle.com/c/higgs-boson)) is used to test the domain transferability of several recent Machine-Learning concepts.

A system utilising relatively recent concepts, such as cyclical learning-rate schedules and data-augmentation, is found to slightly outperform the winning solution of the HiggsML challenge, whilst requiring less than 10% of the training time, no feature engineering, and less specialised hardware. Other recent ideas, such as superconvergence and stochastic weight-averaging are also tested.

Originally presented as an invited talk at the [XIIIth Quark Confinement and the Hadron Spectrum](https://indico.cern.ch/event/648004/) conference at Maynooth University, Ireland, 1-6 August 2018

Data should be present if running via Binder, otherwise it can be downloaded from [here](http://opendata.cern.ch/record/328). Repo includes a copy of my [ML_Tools](https://github.com/GilesStrong/ML_Tools) framework, to ensure future compatability.

## Running
### Locally
1. `git clone https://github.com/GilesStrong/QCHS-2018`
1. `cd QCHS-2018`
1. `mkdir Data`
1. `wget -O Data/atlas-higgs-challenge-2014-v2.csv.gz http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz`
1. `gunzip Data/atlas-higgs-challenge-2014-v2.csv.gz`
1. Either install/update dependencies, or if using conda:
	- `conda env update -f binder/environment.yml`
	- `conda activate QCHS-2018`
1. `python Modules/Data_import.py`

### Binder
Click badge at top, or go to (https://mybinder.org/v2/gh/GilesStrong/QCHS-2018/master). Note, this is pretty slow and shouldn't be used to run the code, only to view it.

### Docker
1. `docker pull gilesstrong/qchs-2018`
1. `docker run -d -p 8888:8888 --name=qchs-2018 gilesstrong/qchs-2018`
1. `docker exec qchs-2018 jupyter notebook list`
1. Click the link, or copy to browser

## References:
- ATLAS collaboration (2014). Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014. CERN Open Data Portal. [DOI:10.7483/OPENDATA.ATLAS.ZBP2.M5T8](http://opendata.cern.ch/record/328)
- Yoshua Bengio, Practical recommendations for gradient-based training of deep architectures, 
- The CMS collaboration, A search using multivariate techniques for a standard model Higgs boson decaying into two photons, CERN, 2012, (https://cds.cern.ch/record/1429931), CoRR, June, 2012, (http://arxiv.org/abs/1206.5533)
- Andre David, Search for the SM Higgs boson decaying in di-photons a case study, (https://indico.lip.pt/event/337/contributions/725/attachments/728/851/171215_Hgg_case_study.pdf)
- Huang et al., Snapshot Ensembles: Train 1, get M for free, CoRR, June, 2017, (http://arxiv.org/abs/1704.00109)
- Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter, Self-Normalizing Neural Networks, CoRR July, 2017, (http://arxiv.org/abs/1706.02515)
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional, NIPS, 2012, (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 
Neural Networks, 
- Ilya Loshchilov and Frank Hutter, SGDR: Stochastic Gradient Descent with Restarts, CoRR, June, 2016, (http://arxiv.org/abs/1608.03983)
- Prajit Ramachandran, Barret Zoph, and Quoc V. Le, Searching for Activation Functions, CoRR, 2017, (http://arxiv.org/abs/1710.05941)
- Leslie N. Smith, No More Pesky Learning Rate Guessing Games, CoRR, June, 2015, (http://arxiv.org/abs/1506.01186)
- Leslie N. Smith and Nicholay Topin, Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates, CoRR, September, 2017, (http://arxiv.org/abs/1708.07120)
- Leslie N. Smith, A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay, CoRR, April, 2018, (http://arxiv.org/abs/1803.09820)


## Acknowledgements
- Some of the callbacks have been adapted from the Pytorch versions implemented in the [Fast.AI library](https://github.com/fastai/fastai), for use with Keras
