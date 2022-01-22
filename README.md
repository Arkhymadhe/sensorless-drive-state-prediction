# Project Title:
### Sensorless Drive State Prediction
___
#### Overview
This project was carried out with the aim to predict the state of sensorless machines based on information as regards their present state of operation.


#### Motivation


#### Quick Start
The dataset for this project was of a size beyond GitHub accommodation levels. As such, an abstraction was provided in order to import the dataset and compress it for storage on the fly. The compressed data file is to be found in the `data/archive` folder. For use in the project, the compressed file is to be decompressed into the `data/dataset
` folder.

To run the scripts, type as below in your CLI:

`py main <--arguments>` e.g.

`py main.py --n_jobs -1`

Acceptable arguments include:
- n_jobs (default = -1)
- visualize (default = False)
- r_state (default = 42; random state)
- data_dir (data directory)
- arch_dir (compressed file directory)
- thresh (minimum limit for feature importance)
- train (create train split?)
- valid (create valid split?)
- test (create test split?)

Others may be found in the `main.py` script.

#### To-Dos

### Citation(s)

