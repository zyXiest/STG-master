# STG-master
An audio-visual question answering (AVQA) framework.


### Requirements
```python
python 3.8
torch 2.4.0
transformers 4.45.2
tensorboard 2.14.0
torchvision 0.19.0
torchaudio 2.4.0
```

## 1. **Download data**
- MUSIC-AVQA: https://gewu-lab.github.io/MUSIC-AVQA/
- MUSIC-AVQA-R: https://github.com/reml-group/MUSIC-AVQA-R
- AVQA Dataset: https://github.com/AlyssaYoung/AVQA
- Activitynet-QA: https://github.com/MILVLG/activitynet-qa

## 2. **Data Preprocessing**

- Feature extraction was performed following the protocol of [TSPM](https://github.com/GeWu-Lab/TSPM), with implementation details available at https://github.com/GeWu-Lab/TSPM.

1. Modify the `configs/stg/parameters.py` file to update dataset path and parameters. 

2. Use the scripts [scripts/extract_clip_feat/*.py](scripts/extract_clip_feat.py) to extract image features and question features.

3. Use the script [scripts/extract_ToMe/extract_tome.py](scripts/extract_ToMe/extract_tome.py) to extract patch features from frames.

    All sample features are collected into HDF5 files (e.g., clip_question_L14.h5, vggish.h5, frame_feat.h5). The processed data is organized as follows:

    ```
    └── data
        ├── MUSIC-AVQA
        │   ├── clip_question_L14.h5
        │   ├── vggish.h5
        │   ├── frame_feat.h5
        │   ├── tomebl14.h5
        ├── └── clip_word_vit_L14.h5
        └── ...
    ```

## 3. **Run codes**

        ```python
        cd STG-master/
        python src/train.py
        ```

## Acknowledgement
The following code was used as a reference for our implementation.
- https://github.com/gewu-lab/pstp-net
- https://github.com/AIM-SKKU/QA-TIGER

We thanks the authors for their efforts.
