
# copulabayesnet

* Load data
* Test Copulas
* Predict from multivariate normal copulas
* Generate several useful plots

## Documentation

[Online documentation](https://copulabayesnet.readthedocs.io/en/latest/ "ReadTheDocs copulabayesnet")

## Installation
Install basic

`pip install copulabayesnet`

If you are looking for latest updates, consider installation directly from sources.

```
git clone https://github.com/SjoerdGn/copulabayesnet.git
cd copulabayesnet
python setup.py install
```

## Functionalities


-- make table here


## Gaussian copula

![Gaussian copula](https://latex.codecogs.com/gif.latex?c_R%5E%7B%5Ctext%7BGa%7D%7D%28u%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%7C%7BR%7D%7C%7D%7D%5Cexp%5Cleft%28-%5Cfrac%7B1%7D%7B2%7D%20%5Cbegin%7Bpmatrix%7D%5CPhi%5E%7B-1%7D%28u_1%29%5C%5C%20%5Cvdots%20%5C%5C%20%5CPhi%5E%7B-1%7D%28u_d%29%5Cend%7Bpmatrix%7D%5ET%20%5Ccdot%20%5Cleft%28R%5E%7B-1%7D-I%5Cright%29%20%5Ccdot%20%5Cbegin%7Bpmatrix%7D%5CPhi%5E%7B-1%7D%28u_1%29%5C%5C%20%5Cvdots%20%5C%5C%20%5CPhi%5E%7B-1%7D%28u_d%29%5Cend%7Bpmatrix%7D%20%5Cright%29%2C)
