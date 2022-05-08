# cs4664_22
Spring '22 ML Capstone Project

- Stock market price prediction application
- Choose between intraday trading, daily trading, or stock-picking algorithms
- Mix between live technical analysis and NLP sentiment analysis
- Will try to get access from Bloomberg terminal, a similar free API, and historical Twitter/article datasets for NLP

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
python3 simulate.py [TODO: arguments]
```