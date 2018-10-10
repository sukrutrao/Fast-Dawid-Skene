# Fast Dawid-Skene
[![Build Status](https://travis-ci.com/GoodDeeds/Fast-Dawid-Skene.svg?branch=master)](https://travis-ci.com/GoodDeeds/Fast-Dawid-Skene)

Implementation of the Fast Dawid-Skene and Hybrid algorithms described in the paper:

Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian. Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment Classification. In Workshop on Issues of Sentiment Discovery and Opinion Mining (WISDOM) at the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2018, August 2018.

[[Paper](http://sentic.net/wisdom2018sinha.pdf)][[arXiv Preprint](https://arxiv.org/abs/1803.02781)][[Code](https://github.com/GoodDeeds/Fast-Dawid-Skene)][[Supplementary Results](https://sites.google.com/view/fast-dawid-skene/supplementary-results)]

Implementations of the Dawid-Skene (Dawid and Skene, 1979) and Majority Voting algorithms are also provided.

These algorithms can be used to aggregate crowd-sourced labels to estimate the true labels. Given the labels of a data point from many annotators out of a set of classes, the algorithms output the most likely correct class for the data point.

## Setup
### Prerequisites
The prerequisites are:
* Python 2.7 or 3.4-3.6
* pip

### Setting up dependencies
All other dependencies can be installed using pip, as
```
$ pip install -r requirements.txt
```

If tests are to be run, use instead,
```
$ pip install -r requirements-dev.txt
```

## Preparing the data
A description of the data format and the procedure to add a new dataset is given [here](data/README.md). A toy dataset is also provided, and can be found [here](data/toy_dataset).

## Running the program
To run the program, use
```
$ python scripts/fast_dawid_skene.py [OPTIONS]
```

To view a list of available options along with descriptions, use
```
$ python scripts/fast_dawid_skene.py --help
```

### Example Run
To run on the toy dataset, with two annotators per question, using the FDS algorithm to obtain predictions, use
```
$ python scripts/fast_dawid_skene.py --dataset toy --k 2 --mode aggregate --algorithm FDS --print_result
```
To run using all available annotations for every question, using the FDS algorithm to obtain predictions, use 
```
$ python scripts/fast_dawid_skene.py --dataset toy --mode aggregate --algorithm FDS --print_result
```

### Running tests
Tests can be run using pytest, as,
```
$ py.test
```

## License
This code is provided under the [MIT License](LICENSE).

Some parts of the code in [this file](fast_dawid_skene/algorithms.py) are derived from [this implementation](https://github.com/dallascard/dawid_skene), and the original license and copyright notice can be found at the top of the file.

## Acknowledgements
Parts of the code for the implementation of the algorithms use or derive from code in [this implementation](https://github.com/dallascard/dawid_skene).

## References

1. A. P. Dawid and A. M. Skene. 1979. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. J. Royal Stat. Soc. Series C 28, 1 (1979), 20–28.
2. Vaibhav B Sinha, Sukrut Rao, Vineeth N Balasubramanian. Fast Dawid-Skene: A Fast Vote Aggregation Scheme for Sentiment Classification. KDD WISDOM 2018.

