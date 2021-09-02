# Holistic Prototype Activation for Few-Shot Segmentation

This repo contains the code for paper "*Holistic Prototype Activation for Few-Shot Segmentation*" by Gong Cheng, Chunbo Lang, and Junwei Han.

### Dependencies

- Python 3.6
- PyTorch 1.3.1
- cuda 9.0
- torchvision 0.4.2
- tensorboardX 2.1

### Datasets

- PASCAL-5i:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
- COCO-20i:  [COCO2014](https://cocodataset.org/#download)

   Please see [OSLSM](https://arxiv.org/abs/1709.03410) and [FWB](https://openaccess.thecvf.com/content_ICCV_2019/html/Nguyen_Feature_Weighting_and_Boosting_for_Few-Shot_Segmentation_ICCV_2019_paper.html) for more details on datasets. 

### Usage

1. Download the prior prototypes of base categories from our [Google Drive](https://drive.google.com/file/d/1YFE-uLBBz89AflsJsa44R-4gqBhhyPer/view?usp=sharing) and put them under `HPA/initmodel/prototypes`. 
2. Download the pre-trained backbones from [here](https://drive.google.com/file/d/1AQcvMHHpURZM67MMgV-S3T0Kz-h2q7FR/view?usp=sharing).
3. Change configuration via the `.yaml` files in `HPA/config`, then run the `.sh` scripts for training and testing.

### Note

Training and testing scripts are coming soon.

### References

This repo is built based on [PFENet](https://github.com/dvlab-research/PFENet) and [DANet](https://github.com/junfu1115/DANet). Thanks for their great work!

### To-Do List

- [x] Support different backbones
- [x] Support various annotations for training/testing
- [ ] Zero-Shot Segmentation (ZSS)
- [ ] FSS-1000 dataset
- [ ] Multi-GPU training



