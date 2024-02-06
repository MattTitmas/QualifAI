# Modelling Formula One Qualifying using Machine Learning
 Cambridge Computer Science Tripos - Part II Project

## Description

This project aims to complete two goals, predicting cutoff times for a given qualifying sessions and predicting the likelihood that a driver will continue into the next session given that they do not complete any more laps of the circuit. This project is completed in collaboration with Mercedes-AMG Petronas Formula One Team.

This is done via Machine Learning techniques, namely a Gaussian Process Regressor, trained on previous races in order to predict statistics about current sessions.

A pipeline is created such that data can be easily extracted from the Mercedes-AMG Petronas F1 Team supplied API and used in a model.

## Getting Started

### Dependencies

* For data extraction:
  * Python 3.8
  * Windows 10
* For model usage:
  * Python 3.8+
  * Windows 10+

### Installing

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements for this project.
```bash
pip install -r requirements.txt
```

### Executing program

#### Data Extraction:
Produces a CSV that the rest of the program can use to create model inputs and expected outputs. Extracts data from the API and processes / filters as necessary.
```bash
python -m DataExtractionProcess.generate_useful_CSV -h
```
Output:
```bash
usage: generate_useful_CSV.py [-h] [-f FILE] -o OUTPUT

Retrieve data for a given race.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Local path of data. (Default=data\all_data.csv)
  -o OUTPUT, --output OUTPUT
                        Output file.
```
#### Cutoff Time Prediction:
Takes a CSV and generates a prediction for the cutoff time for a specified session. Assumes that part of that session will be contained within the supplied CSV.
```bash
python -m ModelUsage.predict_cutoff -h
```
Output:
```bash
usage: predict_cutoff.py [-h] -c CSV -s SESSION -d DATE [-t TYRES] [-tt TRAIN_TO] [-rs] [-rc] [-rl] [-pb]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV, --csv CSV     Location of the CSV file.
  -s SESSION, --session SESSION
                        Current session.
  -d DATE, --date DATE  Date of the current event (as CSV shows).
  -t TYRES, --tyres TYRES
                        Most likely tyre choice for all drivers (default="Soft").
  -tt TRAIN_TO, --train_to TRAIN_TO
                        How many rows of the CSV to use as training data.
  -rs, --return_std     Return a corresponding standard deviation.
  -rc, --return_conf    Return a corresponding confidence interval.
  -rl, --return_loss    Return the loss of the kernel.
  -pb, --progress_bar   Show a progress bar.
```

#### Should the driver remain in the garage:
Takes a CSV and generates a prediction whether the driver should remain in the garage or set another lap.
```bash
python -m ModelUsage.remain_in_garage -h
```
Output:
```bash
usage: remain_in_garage.py [-h] -c CSV -s SESSION -d DATE --drivers DRIVERS [DRIVERS ...] [-t TYRES] [-tt TRAIN_TO] [-rs] [-rc] [-rl] [-pb]

optional arguments:
  -h, --help            show this help message and exit
  -c CSV, --csv CSV     Location of the CSV file.
  -s SESSION, --session SESSION
                        Current session.
  -d DATE, --date DATE  Date of the current event (as CSV shows).
  --drivers DRIVERS [DRIVERS ...]
                        Drivers to return results for.
  -t TYRES, --tyres TYRES
                        Most likely tyre choice for all drivers (default="Soft").
  -tt TRAIN_TO, --train_to TRAIN_TO
                        How many rows of the CSV to use as training data.
  -rs, --return_std     Return a corresponding standard deviation.
  -rc, --return_conf    Return a corresponding confidence interval.
  -rl, --return_loss    Return the loss of the kernel.
  -pb, --progress_bar   Show a progress bar.

  -pb, --progress_bar   Show a progress bar.
```


## Authors

Author. [Matthew Titmas](https://github.com/MattTitmas)
## Version History

* 0.1
    * Initial Release


## License

This project is licensed under the MIT License - see the LICENSE file for details
