# quickq
Prediction of partition functions using trained ML estimators.
![logo](.quickqlogo.png)

This repository consitutes the final products of the works: [DOI]

## Installation
To use the tool, install and activate the conda environment, and then install the package.

```
conda create --file environment.yml --name <new_env_name>
conda activate <new_env_name>
pip install .
```
## Usage brief
Once installed, the package can be used at the command line as `python quickq.py <arguments>`:

```
usage: quickq.py [-h] (-q | -t | -d) root

Predict partition functions.

positional arguments:
  root          Path to root directory containing structure files or reaction directories.

optional arguments:
  -h, --help    show this help message and exit
  -q, --qest    Use Qest to predict partition functions of molecules
  -t, --qests   Use QesTS to predict partition functions of unknown transition states.
  -d, --double  Use Qest and QesTS to predict partition functions of unknown transition states.
```
## Usage details
Note that structures must contain only carbon, hydrogen, nitrogen and oxygen. The predictor will function for systems with more than 7 C, N, O atoms due to the size independance of the EncodedBonds featurizer [1], however it was not tested for these systems. Input data must strictly adhere to the below format for predictions to be made.

### 1. Qest Usage
Qest is a predictor of partition functions for arbitrary structures. To do so, two files are required for each system:
- an extended XYZ file with the extension `.extxyz` containing atom types and positions
- a comma seperated value file with extension `.csv` containing the first column as temperatures in Kelvin to predict the partition function at. This column **must either have label "T" or start with "T "**. Other columns will not be modified.

These files must have the same file header, eg `molecule1.extxyz` and `molecule1.csv`.

Place these files for each system alone in a directory whose path we call `root`.

The directory structure should then look like, where `XX` and `YY` represent an arbitrary names, note that no name should be repeated between systems:
```
/root/
|-XX.extxyz
|-XX.csv
|-YY.extxyz
|-YY.csv
|-...
```

The predictions can then be executed by the following command `python quickq.py <root> -q`, after which each csv file will have a new column `log_qpart_predicted` corresponding to the natural log of the predicted partition functions.

### 2. QesTS Usage
QesTS is a predictor of unknown transition state partition functions. Four files are required for each reaction, where `XX` is an arbitrary name:
- an extended XYZ file with the name `rXX.extxyz` containing atom types and positions **for the reactant**
- an extended XYZ file with the name `pXX.extxyz` containing atom types and positions **for the product**
- a comma seperated value file with name `rXX.csv` containing at least two columns. One column is a column of temperatures in Kelvin and it must be in the first position. This column **must either have label "T" or start with "T "**. The second column must have the label "log_qpart" with values of the natural logarithm of the reactant partition functions at the specified temperatures. Other columns will not be modified.
- a comma seperated value file with name `pXX.csv` with the same format as `rXX.csv` except with product logged partition functions.

**Note that the temperatures in the two csv files must be identical. This will not be checked.**

Place these files alone in a directory entitled "rxnXX". This directory represents the reaction with identifier XX. Place as many reactions as interested in alone in a directory whose path we call `root`.
The directory structure should then look like, where `XX` and `YY` represent a arbitrary names, note that no name should be repeated between systems:
```
/root/
|-rxnXX/
| |-rXX.extxyz
| |-pXX.extxyz
| |-rXX.csv
| |-pXX.csv
|-rxnYY/
| |-rYY.extxyz
| |-pYY.extxyz
| |-rYY.csv
| |-pYY.csv
...
```
The predictions can then be executed by the following command `python quickq.py <root> -t`, after which each reaction directory will contain a csv file entitled "tsXX.csv" where XX is that reactions identifier. This csv file contains the temperatures used to predict the partition function, and a column entitled "log_qpart_predicted" associated with the logarithm of the partition functions of the unknown transition state at those temperatures.

### 3. Double Usage
Double prediction utilizes Qest and QesTS to predict partition functions of unknown transition states using only structure and temperature. The directory structure required is identical to QesTS prediction (See **2. QesTS Usage**) except that _the "log_qpart"_ columns in the reactant and product csv files are no longer ncessary. Note that the temperatures column still must be present.

The predictions can then be executed by the following command `python quickq.py <root> -d`, after which each reaction directory will contain a csv file entitled "tsXX.csv" where XX is that reactions identifier. This csv file contains the temperatures used to predict the partition function, and a column entitled "log_qpart_predicted" associated with the logarithm of the partition functions of the unknown transition state at those temperatures.

## Toy data
The repository contains a directory "toy_data" containing three datasets "qest_test", "qests_test" and "double_test". These datasets each contain the minumum amount of information needed to make predictions (not that the extra information in the extended xyz files are not necessary, only the atoms and positions) using each of the three models.  

## References
[1] C. R. Collins, G. J. Gordon, O. A. Von Lilienfeld, and D. J. Yaron, J. Chem. Phys. 148, 241718 (2018).
