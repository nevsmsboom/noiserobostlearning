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
Do the first-phase training in ours.py and save relative npz files to folder linearregression/alldata, with filenames like experimentnameyoulike_datasetname_noiserate.npz.
Run lrtrain.py to generate linear regressors, and they will be saved with the relavant dataset name in linearregression folder.

#### Run the main script 
MedMNIST dataset can be placed at ../../medmnistdata, or can be modified through the dataloader.

## Team

Maolin Li, Giacomo Tarroni


## License

Improving synthetic anomaly based out-of-distribution with harder anomalies is relased under the MIT License.




