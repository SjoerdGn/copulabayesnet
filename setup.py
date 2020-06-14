from setuptools import setup, find_packages
import io

readme = io.open('README.rst', encoding='utf-8').read()

requirements = ['numpy>=1.16.0', 
                'pycopula', 
                'scipy', 
                'statsmodels',
                'matplotlib', 
                'scikit-gof']

setup(name='copulabayesnet', 
      author='Sjoerd Gnodde',
      url = 'https://github.com/SjoerdGn/copulabayesnet',
      version = '0.1.0', 
      description = 'Test different copulas and use multivariate Gaussian copulas',
      long_description = readme,
      install_requires=requirements,
      packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
        )),
      package_dir = {'':'src'},
      python_requires='>=3.6')