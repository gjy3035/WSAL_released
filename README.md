# WSAL_released

This is an official implementation of the paper "[Weakly Supervised Adversarial Domain Adaptation for Semantic Segmentation in Urban Scenes](https://arxiv.org/abs/1904.09092)" (completed in **October 2017**, accepted by IEEE TIP in **March 2019**).

We plan to finish ```readme.md``` in **December 2019**. 

## Some Essential Information:

- Pytorch 0.2
  - This work is completed in **October 2017**. We have no plan to upgrade it to a newer Pytorch version because of limited time.
- Python2.7
- TensorboardX


## Simple Instruction:
- 1 Download GTA 5, SYN, Cityscapes datasets.
- 2 Generate object-level labels:
  - 2.1 Detect the foreground objects using [DSOD](https://github.com/szq0214/DSOD).
  - 2.2 Get the background objects' locations according to segmentation masks.
  - 2.3 Merge the locations of fore/background objects.
  - 2.4 Generate SSD-style detection labels, a txt file. Each line contains ```filename number_of_objects xmin ymin xmax ymax ...```. 
   Example:
   ```
   1.png 1 0 0 1 1 
   2.png 2 0 0 1 1 2 2 4 4
   ...
   ```
- 3 Make ROI pooling.
- 4 Train model: ```python run_WASAL.py```.
