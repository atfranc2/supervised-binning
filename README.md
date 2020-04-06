# Supervised Binning
A Python class for binning continuous variables in a way that the bins significantly predict a binary target variable

# Overview
Intial characteristic analysis is a binning method that bins continuous predictor variables into disctrete categroical bins. This is useful when implementing a logistic regression model, but have variables that are not linearly associated with the logit. 

Initial characteristic analysis starts by breaking the continuos variable into a pre-specifed number of bins with equal sample sizes across bins. Fishers exact test is used to determine if a significant difference bewtween adjacent bins exists. If no significant exists then the bin contents are merged and the algorithm starts from the beginning. This process is repeated until all bins are significantly different. 

# Methods

fit(self, x, y, alpha = 0.05, start_bins = 200, disp = True)

transform(self, x)

fit_transform(self, x, y, alpha = 0.05, start_bins = 200, disp = True)

