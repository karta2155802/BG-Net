# Boundary-Guiding Interactive Hand Object Pose Estimation

## Requirement
* Pytorch2.0
* CUDA 11.8
* Other packages
```
pip install -r requirements.txt
```

### Directory
```
datasets
|-- HO3D
|   |-- data
|   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |-- evaluation
|   |   |-- train_segLable
|   |   |-- ho3d_train_data.json
|   |   |-- train.txt
|   |   |-- evaluation.txt
|-- DEX_YCB
|   |-- 20200709-subject-01
|   |-- ......
|   |-- object_render
|   |   |-- train_segLable
|   |-- dex_ycb_s0_train_data.json
|   |-- dex_ycb_s0_test_data.json
|-- YCB_Video_Models
|   |-- dex_simple
${ROOT}  
|-- assets
|   |-- mano_models
    |-- sealed_faces.npy
|-- dataset
|-- exp-results
|-- mano
|-- manopth
|-- networks
|-- utils
|-- model.py
|-- traineval.py
```

### Data  
You need to follow directory structure of the `data` as below.  

* Download HO3D(version 2) data [data](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/)
* Download DexYCB data [data](https://dex-ycb.github.io/)
* Download the process data [data](https://drive.google.com/drive/folders/1QyRvGCXKX3suIIUvv6EQ1FZwG050evY0?usp=drive_link)
* Download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from [here](https://drive.google.com/drive/folders/1QyRvGCXKX3suIIUvv6EQ1FZwG050evY0?usp=drive_link) and place at `assets/mano_models`.
  
### Pytorch MANO layer (Already included in this repo)
* For the MANO layer, I used [manopth](https://github.com/hassony2/manopth). 

### Train  
#### HO3d
```
python traineval.py --use_ho3d
```
#### Dex-ycb
```
python traineval.py
```
### Test  
#### HO3d
```
python traineval.py --use_ho3d --evaluate
```
#### Dex-ycb
```
python traineval.py --evlaute
```  

## Acknowledgments
We thank: 
* [Semi-Hand-Object](https://github.com/stevenlsw/Semi-Hand-Object.git) 
* [HandOccNet](https://github.com/namepllet/HandOccNet.git)
* [HFL-Net](https://github.com/lzfff12/HFL-Net)


