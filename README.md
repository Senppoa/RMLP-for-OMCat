# RMLP for Organometallic Catalysis

![License](https://img.shields.io/github/license/Senppoa/RMLP-for-OMCat) [![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jctc.5c01047-137446)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01047)

## Overview

This repository includes the initial guess transition state structures of the organometallic catalytic reaction (Rh-catalyzed ethylene hydrogenation) discussed in the article [***Accelerating Transition State Search and Ligand Screening for Organometallic Catalysis with Reactive Machine Learning Potential***](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01047), as well as the corresponding code for transition state structure optimizations and IRC calculations driven by the reactive machine learning potential (RMLP) model.

## Repo Contents

The contents of each folder are as follows:

1. **dataset**: The transition state initial guess structures of the organometallic catalytic reaction systems in the paper (including the organic ligands designed by `ScaffoldCAMD`, metal rhodium, and reaction substrates).
2. **example**: Workfolder of transition state optimization by RMLP model.
3. **models**: The RMLP models in the article, including MACE w/ NMS, AL MACE w/ NMS, MACE w/o NMS, PaiNN w/ NMS.
4. **neuralneb**: Dependency modules of PaiNN model (sourced from [NeuralNEB](https://gitlab.com/matschreiner/neuralneb)).
5. **ts_opt.py**: Script for transition state optimization driven by RMLP models.

## Required modules

- torch 2.5.1+cu121
- numpy 1.26.4
- xtb 22.1
- mace-torch 0.3.6
- ase 3.23.0
- sella 2.3.4
- matplotlib 3.8.2
- natsort 8.4.0
- x3dase 1.1.4

## Usage tutorial

After downloading the repository using git clone or similar commands, move to the generated directory and run the following:

```
python ts_opt.py --model_name='mace_nms'
```

This command will use the MACE with NMS model to optimize the initial guess of transition state structures in `./example/input` folder , and output to `./example/output`.

Other arguments:

```
--input_path
```

Type: str. Specifies the XYZ format geometry path for the input.

```
--output_path
```

Type: str. Specifies the output file path.

```
--model_path
```

Type: str. Specifies the RMLP model file path.

## Contact

Please contact us ([liuqilei@dlut.edu.cn](mailto:liuqilei@dlut.edu.cn)) if you have any question about our implementation.

