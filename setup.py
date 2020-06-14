from setuptools import setup, find_packages
import io

readme = io.open('README.md', encoding='utf-8').read()

requirements = ['numpy>=1.16.0', 
                'pycopula', 
                'scipy', 
                'statsmodels',
                'matplotlib', 
                'scikit-gof',
                'hydroeval',
                'easygui',
                'biokit']

setup(name='copulabayesnet', 
      packages = ['copulabayesnet'],
      author='Sjoerd Gnodde',
      author_email='',
      license='MIT',
      url = 'https://github.com/SjoerdGn/copulabayesnet',
      keywords = ['Copulas', 'Bayesian Networks', 'Multivariate normal'],
      version = '0.1.0', 
      description = 'Test different copulas and use multivariate Gaussian copulas',
      long_description = readme,
      install_requires=requirements,
      packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
        )),
      package_dir = {'':'src'},
      python_requires='>=3.6',
      classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ])