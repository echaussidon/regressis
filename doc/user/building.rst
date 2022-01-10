.. _user-building:

Building
========

Requirements
------------
Strict requirements are:

  - scikit-learn (and dependencies)
  - fitsio
  - healpy
  - astropy

pip
---
To install **regressis**, simply run::

  python -m pip install git+https://github.com/echaussidon/regressis

git
---
First::

  git clone https://github.com/echaussidon/regressis.git

If you have an ssh-key and want to contribute, you should prefer::

  git clone git@github.com:echaussidon/regressis.git

To install the code::

  python setup.py install --user

Or in development mode (any change to Python code will take place immediately)::

  python setup.py develop --user
