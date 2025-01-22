# RMLP for organometallic catalysis

## Introduction

This repository includes the structures of organometallic catalytic reaction systems discussed in the article "**Accelerating the Transition State Search of Organometallic Catalysis with Reactive Machine Learning Potentials**," as well as the corresponding code for transition state structure optimization using the integrated reactive machine learning potential (RMLP) model.

The contents of each folder are as follows:

1. **dataset**: The transition state initial guess structures of the organometallic catalytic reaction systems in the paper (including the organic ligands designed by ScaffoldCAMD, metal rhodium, and reaction substrates).
2. **example**: Workfolder of transition state optimization by RMLP model.
3. **models**: The RMLP models in the article, including MACE w/ NMS, MACE w/o NMS, PaiNN w/ NMS.
4. **neuralneb**: Dependency modules of PaiNN model.
5. **ts_opt.py**: Script for transition state optimization driven by RMLP models.

## Required modules

- torch
- numpy
- xtb
- mace-torch
- ase
- sella
- matplotlib
- natsort
- configparser
- os
- time
- traceback
- argparse
- x3dase

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

