# SparseIndex
Code for the sparse index tracking problem.
We take a user constructed index of stock symbols, and train a model on the return matrix of the stocks in the index.
over a given date range.
  The goal is to compute a portfolio consisting of a sparse subset of the stocks in the index that will track the index
  with small tracking error.  That is, we're trying to find a quick approximation of the problem:
  
  min_w ||Rw - y||_2 + \lambda*card(w)
  subject to w \geq 0, \sum_i w_i = 1
  
 where 
 R is an mxn matrix of returns.  n is the number of assets in the index.  m the time periods of the training set
 So the ith column is the vector of returns for the ith asset.
 
 w is a length n vector of weights.  w_i represents the proportion of the portfolio invested in asset i.
 
 y is a length m vector representing the returns for the composit index over the training period
 
 card(w) maps w to its cardinality.
 
 lambda is a regularization parameter that penalizes the cardinality of the solution. 
 
 The constraint w \geq 0 means we do not allow short selling
 \sum_i w_i =1 is a normalization constraint (We have $1 to invest, and we're investing it all).
