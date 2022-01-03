from setuptools import setup


setup(name='regressis',
      version='0.0.1',
      author='Edmond Chaussidon, Arnaud de Mattia',
      author_email='',
      description='Regression of target density against observational condition templates',
      license='GPLv3',
      url='http://github.com/echaussidon/regressis',
      install_requires=['scikit-learn', 'fitsio', 'healpy', 'astropy'],
      packages=['regressis']
)
