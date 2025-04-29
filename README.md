# QRSAC
Implementation of Quantile Regression Soft Actor Critic (QRSAC) from "Outracing champion Gran Turismo drivers with deep reinforcement learning" by Peter R. Wurman, Samuel Barrett, Kenta Kawamoto, James MacGlashan, Kaushik Subramanian, Thomas J. Walsh, Roberto Capobianco, Alisa Devlic, Franziska Eckert, Florian Fuchs, Leilani Gilpin, Piyush Khandelwal, Varun Kompella, HaoChih Lin, Patrick MacAlpine, Declan Oller, Takuma Seno, Craig Sherstan, Michael D. Thomure, Houmehr Aghabozorgi, Leon Barrett, Rory Douglas, Dion Whitehead, Peter Dürr, Peter Stone, Michael Spranger & Hiroaki Kitano. [[Paper]](https://www.nature.com/articles/s41586-021-04357-7). 

This repository is based on [RLkit](https://github.com/vitchyr/rlkit) and [DSAC](https://github.com/xtma/dsac), two popular reinforcement learning frameworks implemented in PyTorch.

This repository is also heavily adapted from [[Code]](https://github.com/shilpa2301/QRSAC)

## Requirements
- python 3.10+
- All dependencies are available in requirements.txt and environment.yml

## Usage
To run donkeycar you can write your experiment settings in configs/donkeycar.yaml and run with 
```
python qrsac.py --config your_config.yaml --gpu 0 --seed 0
```
Set `--gpu -1`, your program will run on CPU.

Set `--mode curriculum`, your program will use the curriculum learning framework

This approach uses an autoencoder which can be found under `/logs/ae-32_1745884521_best.pkl`

## Autoencoder Setup
To train your own autoencoder to then use for training first run
```
python record_data_ae.py -f {your desired save folder}
```
Then to train the autoencoder run
```
python -m ae.train_ae --n-epochs 400 --batch-size 8 --z-size 32 -f {folder where you saved from first command} --verbose 1
```
To use you autoencoder replace `ae_path` in `configs/donkeycar.yaml` with `logs/your_ae_file`
## Experiments
Two different experiments are run to determine if the curriculum learning approach is more sample efficient

To test the final policy for the standard (non-curriculum learning) approach run
```
python eval_script.py --checkpoint './eval_models/StandardParams.pkl'
```

To test the final policy for the curriculum learning approach run
```
python eval_script.py --checkpoint './eval_models/CurriculumParams.pkl'
```



