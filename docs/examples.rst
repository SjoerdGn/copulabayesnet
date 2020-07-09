Examples
========

Predict with the multivariate normal method
-------------------------------------------

In this example, a k-fold test with a Bayesian network is done.

Load the package
::

  from copulabayesnet import bncopula as bc
  import numpy as np
  from copulabayesnet.data_preprocessor import CorrMatrix


Load dataset. In this case, the standard diabetes dataset of `sklearn <http://www.python.org/>`_ is used.
::

  from sklearn.datasets import load_diabetes
  X, y = load_diabetes(return_X_y=True)
  X = X.T # bncopula requires separate rows with the variables

Let's only use the actual continuous variables
::

  X = np.delete(X, (1), axis = 0)
  data = np.vstack([X, y])


We use pandas to handle the data in this case.
::

  import pandas as pd

Add some fake titles and make a DataFrame
::

  titles = 'abcdefghkt'

  datadict = {}
  for i in range(len(data)):
      datadict[titles[i]] = data[i]
  datadf = pd.DataFrame(datadict)

If you want to create a file, set to ``True``.
::

   if False:
        datadf.to_csv("../../data/copulabayesnet/diabetes.csv", index = False)

Now make a matrix in Uninet. Make sure to add the variables in the correct order to the model.
Then, do the testing:

Load the matrix.
::

  cm = CorrMatrix("example_matrix.txt")

Create a predict object.
::

  pred = bc.Predict(data, [9], R = cm.R)

5-fold test with the Gaussian mixture model (``mixed gaussians  = 'mg'``)
::
  nses, kges = pred.k_fold_bn(fit_func = 'mg', n = 500, numpars = 3, k = 5)
  print()
  print("The NSEs are: ", nses)
  print()
  print("The KGEs are [kge, rho, alpha, beta]: ", kges)

Which returns
::

  >>Calculating took 5.144308805465698 seconds
  >>Calculating took 5.216999769210815 seconds
  >>Calculating took 22.19534945487976 seconds
  >>Calculating took 5.084670782089233 seconds
  >>Calculating took 4.949423551559448 seconds
  >>
  >>The NSEs are:  [0.34766019343269194, 0.4524929711759399, 0.48652117173319775, 0.3948568188728876, 0.22643399304608447]
  >>
  >>The KGEs are [kge, rho, alpha, beta]:  [array([0.60711605, 0.63257781, 0.86112566, 0.99147643]), array([0.66501358, 0.69589118, 0.86273868, 0.97011597]), array([0.62806341, 0.70763917, 0.77671746, 1.05483499]), array([0.55459502, 0.63709045, 0.74180172, 1.00398893]), array([0.54392231, 0.56460633, 0.86556329, 1.01913055])]

See also: `<https://github.com/SjoerdGn/copulabayesnet/blob/master/examples/test_bayesian_network.py>`_
