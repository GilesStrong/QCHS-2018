# Kaggle_HiggsML
Repo for my solution to the [HiggsML Kaggle challenge](https://www.kaggle.com/c/higgs-boson/).

## Installing
Developed in Python 2.7. Requires methods collected in [ML_Tools](https://github.com/GilesStrong/ML_Tools). Download, checkout relevent tag, and add base directory to PYTHONPATH.

Latest tags:
- Kaggle_HiggsML - [Blog2](https://github.com/GilesStrong/Kaggle_HiggsML/tree/Blog2)
- ML_Tools - [HiggsMLState2](https://github.com/GilesStrong/ML_Tools/tree/HiggsMLState2)
- Related [blog post](https://amva4newphysics.wordpress.com/2018/04/26/train-time-test-time-data-augmentation/)

Previous tags:
- Kaggle_HiggsML - No tag used and compatability not ensured (lots of changes over the days)
- ML_Tools - [HiggsMLState](https://github.com/GilesStrong/ML_Tools/tree/HiggsMLState)
- Related [blog post](https://amva4newphysics.wordpress.com/2018/03/21/higgs-hacking/)

## Running
1) Download [data](https://www.kaggle.com/c/higgs-boson/data) to `Kaggle_HiggsML/Data`.
1) Import and preprocess data using `Kaggle_HiggsML/Data/1_Data_Import.ipynb`. Can alter settings to adjust preprocessing.
1) Run classifier, e.g. `Kaggle_HiggsML/Classifiers/Day10/Day_10_RotRefAugTrains.ipynb`. N.B. `dirLoc` may need to be adjusted to point to `Kaggle_HiggsML/Data/`
