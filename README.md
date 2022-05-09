# cs4664_22
Spring '22 ML Capstone Project

- Stock market price prediction application
- Choose between intraday trading, daily trading, or stock-picking algorithms
- Mix between live technical analysis and NLP sentiment analysis
- Will try to get access from Bloomberg terminal, a similar free API, and historical Twitter/article datasets for NLP

- ARIMA: A notebook that allows you to run the arima model on NFLX data
- DATA: Data used throughout the project

- \* DATA_APRIL: Data used exclusively for the final test and the decision graphs
- \* LSTM-CNN: Our final model
       run python lstm.py to run the model
       
- LSTM_TCN: Our LSTM and TCN Models
- MA: Our simple moving average script
- MACD: Our simple MACD script
- IMAGES: Some stored images from our report
- MODELS: An interface for testing different models with consistency
- nlpZOE: An NLP model for NFLX
- utils: misc
- api_requests.py: How to request data. Should probably remove my key
- plot.py: Script for making some graphs


## Prerequisites

* `python`
* `pip`
* `venv`
* `git lfs`

Example install script:
```bash
sudo apt-get update
sudo apt-get install libpython3-dev python3-pip python3-venv git-lfs
git lfs install && git lfs pull
```

## Installation

Setup `python` environment:
```bash
python3 -m venv CS4664-env --without-pip --system-site-packages
source ./CS4664-env/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 driver.py [TODO: arguments]
```
