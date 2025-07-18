# RMLP for organometallic catalysis

## Introduction

This repository includes the structures of organometallic catalytic reaction systems discussed in the article "**Accelerating the Transition State Search of Organometallic Catalysis with Reactive Machine Learning Potentials**," as well as the corresponding code for transition state structure optimization using the integrated reactive machine learning potential (RMLP) model.

The contents of each folder are as follows:

1. **dataset**: The transition state initial guess structures of the organometallic catalytic reaction systems in the paper (including the organic ligands designed by ScaffoldCAMD, metal rhodium, and reaction substrates).
2. **example**: Workfolder of transition state optimization by RMLP model.
3. **models**: The RMLP models in the article, including MACE w/ NMS, AL MACE w/ NMS, MACE w/o NMS, PaiNN w/ NMS.
4. **neuralneb**: Dependency modules of PaiNN model.
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

