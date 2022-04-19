import os
import sys

from setuptools import setup


package_basename = 'regressis'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='Edmond Chaussidon, Arnaud de Mattia',
      author_email='',
      description='Regression of target density against observational condition templates',
      license='GPLv3',
      url='http://github.com/echaussidon/regressis',
      install_requires=['scikit-learn', 'fitsio', 'healpy', 'astropy'],
      package_data={package_basename: ['*.mplstyle', 'data/*']},
      packages=[package_basename]
)
