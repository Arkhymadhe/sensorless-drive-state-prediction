# Project Title:
## Sensorless Drive State Prediction
___

### Overview
This project was carried out with the aim to predict the state of sensorless machines based on information as regards 
their present state of operation.

___
### Motivation
___
###### _Background_

A lot of the time, the manufacturing processes that allow humans to live the comfortable lives we do cannot be carried 
out directly by ourselves. This might be due to speed and/or efficiency limits. Other times, they may be due to outright
inability to do so. For instance, the fractional distillation of crude oil into petroleum products. Ths is not a process
that we carry out directly, due to the hostile conditions. This is where our machines and machinery come in.

The machines and machinery are made of metal and engineered to be able to survive in hostile environments, for instance 
the possibly hot and corrosive environments in a fractionating column at an oil refinery plant. Dedicated machines are 
mounted in the environment within which the process of interest is occurring, and the machines are mounted with 
appropriate sensors. While the machines carry out their designed and assigned tasks, the sensors report on the state 
of the machine. This allows us to monitor the status of the machine, hence being able to extract it from the environment
for required repairs and servicing based on information derived from the sensors.

In essence, the sensor is the watcher. But a few questions are pertinent:
- Who shall watch the watcher?
- What happens when the sensor itself breaks down, or worse, begins to malfunction, thereby returning inaccurate 
readings?
- In the event that the sensor breaks down, it is at least obvious, since there is a break in the stream of diagnostic 
signals received. But what happens when the signals being received are anomalous in nature?
- Worse, what happens when the readings are not necessarily anomalous, just inaccurate? In most cases, this might be 
the case, with the readings being inaccurate but not anomalous. Yet, inaccurate reading are oft enough to deceive us 
as regards the state of our machinery.


###### _Crux_

One answer to this problem is to eliminate the need for sensors entirely. This makes sense as:
- Sensors are almost always in operation, hence having relatively short lifespans.
- Require frequent and regular replacement.
- Can experience signal interference.

The idea behind this replacement is to observe the machines directly, without the intermediary of a sensor.
 As the machines depend on electricity to function, measuring the properties of the electric signal being transmitted to
 the machines will allow good diagnostics of the machine state, hence the advent of `serverless machines`.

___
### Data Understanding
The dataset possess `58509` records, with `48` fields and `1` target (dependent) variable. The features are extracted 
from electric current drive signals from a motor with both intact and defective components. 
This results in 11 different classes (for our dependent variable) with different conditions. Each condition has been 
measured several times by 11 different operating conditions, this means by different speeds, 
load moments and load forces. The current signals are measured with a current probe and an oscilloscope on two phases.

Each record belongs to one of 11 classes:

- `1` (Healthy)
- `2 - 11` (The degree of fault present)


##### _Aim_

The aim of this project is to be able to predict the state of the machine based on readings obtained from the electric 
current upon which the machine runs. This means that we can mount our sensors on the power source, and based on the 
power source, we can tell the status of the machines that run on said source.

___
### Quick Start
The dataset for this project was of a file size beyond GitHub accommodation levels. As such, an abstraction is provided 
to read the dataset into memory and compress it for storage on the fly. The compressed data file is to be found in the 
`data/archive` directory. For use in the project, the compressed file is to be decompressed into the `data/dataset` 
directory.

To run the scripts, type as below in the Terminal:

1. Navigate to the `scripts` directory.

```bash
$ cd scripts
```
2. Next, run the `main.py` file with the following syntax:

    `py main --argument argument_value`

Example:

```bash
$ py main.py --n_jobs -1
```
Acceptable arguments include:
- `n_jobs` (default = -1)
- `visualize` (default = False)
- `r_state` (default = 42; random state)
- `data_dir` (data directory)
- `arch_dir` (compressed file directory)
- `thresh` (minimum limit for feature importance)
- `train` (create train split?)
- `valid` (create valid split?)
- `test` (create test split?)

Others may be found in the `main.py` script.

3. Generated diagnostics, text and images, will populate the `reports/text` and `reports/images` directories 
respectively.
4. Find trained model artefact in the `artefacts` directory.

___
### Performance Report
Feature selection techniques were applied. This saw a considerable performance improvement from `~ 85 %` to `~ 94 %`.
After training, a performance of `~ 96 %` was recorded across board for the major classification metrics (accuracy, 
f1-score, recall, and precision), via `macro` averaging.

The `ExtraTreesClassifier` algorithm provided the base learner for the final `AdaBoost` classifier.

___
### To-Dos
___

### Citation(s)
1. J. Holtz, "[Sensorless Control of Induction Machines—With or Without Signal Injection?]
(https://ieeexplore.ieee.org/document/1589362)," in IEEE Transactions on Industrial Electronics, vol. 53, no. 1, 
pp. 7-30, Feb. 2006, doi: 10.1109/TIE.2005.862324.