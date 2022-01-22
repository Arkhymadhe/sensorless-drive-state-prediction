# Project Title:
### Sensorless Drive State Prediction
___
#### Overview
This project was carried out with the aim to predict the state of sensorless machines based on information as regards their present state of operation.


#### Motivation


#### Quick Start
The dataset for this project was of a file size beyond GitHub accommodation levels. As such, an abstraction is provided to read the dataset into memory and compress it for storage on the fly. The compressed data file is to be found in the `data/archive` directory. For use in the project, the compressed file is to be decompressed into the `data/dataset
` directory.

To run the scripts, type as below in the Terminal:
1. Navigate to the `scripts` directory.
```
./ $ cd scripts
```
2. Next, run the `main.py` file with the following syntax:

    `py main --argument argument_value`

Example:

```
./scripts/ $ py main.py --n_jobs -1
```
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
3. Generated diagnostics, text and images, will populate the `reports/text` and `reports/images` directories respectively.
4. Find trained model artefact in the `artefacts` directory.


#### Performance Report
Feature selection techniques were applied. This saw a considerable performance improvement from `~ 85 %` to `~ 94 %`.
After training, a performance of `~ 96 %` was recorded across board for the major classification metrics (accuracy, f1-score, recall, and precision), via `macro` averaging.

An `ExtraTreesClassifier` provided the base algorithm for the final `AdaBoost` classifier.



#### To-Dos

### Citation(s)

