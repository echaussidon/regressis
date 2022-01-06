# Regressis

**regressis** (regression of spectroscopic and imaging systematics) is a package to
regress variations of target/galaxy density against a set of templates accounting for observational conditions.

## Requirements

Strict requirements are:

  - scikit-learn (and dependencies)
  - fitsio
  - healpy
  - astropy

## Documentations

Documentation is hosted on Read the Docs, [regressis docs](https://regressis.readthedocs.io/).

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/echaussidon/regressis
```

### git

First (if you do not use a ssh-key):
```
git clone https://github.com/echaussidon/regressis.git
```
Or (if you use a ssh-key):
```
git clone git@github.com:echaussidon/regressis.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
python setup.py develop --user
```

## License

**regressis** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/echaussidon/regressis/blob/main/LICENSE).
