## Combating Label Noise in Medical Image Classification: A Unified Approach with Noise Rate Estimation and Sparse Regularization

Implementation of manuscript "Combating Label Noise in Medical Image Classification: A Unified Approach with Noise Rate Estimation and Sparse Regularization"


## Citing 
If you use this implementation, please cite our [paper](https://arxiv.org/abs/2308.01412):
```
@misc{li2024sampleselectionnoiserate,
      title={Sample selection with noise rate estimation in noise learning of medical image analysis}, 
      author={Maolin Li and Giacomo Tarroni},
      year={2024},
      eprint={2312.15233},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2312.15233}, 
}
```


## How to use

#### Train the linear regression-based noise rate estimation module
First, run ours.py to perform the initial training phase. Save the resulting .npz files to the linearregression/alldata/ directory, with filenames like experimentnameyoulike_datasetname_noiserate.npz.
Next, execute lrtrain.py to train the linear regressors. The trained models will be saved in the linearregression/ folder, named according to their corresponding dataset.

#### Create NoisyCXR dataset 
Run makebinarynoisycxr.py in makeNoisyCXRdataset folder. This script will generate a dictionary that maps image files to their corresponding clean and noisy labels.

#### Run the main script 
Ensure the MedMNIST dataset is located at ../../medmnistdata. If you have placed it in a different directory, you will need to update the path in the dataloader.

## Team

Maolin Li, Giacomo Tarroni


## License

Improving synthetic anomaly based out-of-distribution with harder anomalies is relased under the MIT License.




