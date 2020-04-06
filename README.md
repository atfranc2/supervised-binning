# Supervised Binning
A Python class for binning continuous variables in a way that the bins significantly predict a binary target variable

# Overview
Intial characteristic analysis is a binning method that bins continuous predictor variables into disctrete categroical bins. This is useful when implementing a logistic regression model, but have variables that are not linearly associated with the logit. 

Initial characteristic analysis starts by breaking the continuos variable into a pre-specifed number of bins with equal sample sizes across bins. Fishers exact test is used to determine if a significant difference bewtween adjacent bins exists. If no significant exists then the bin contents are merged and the algorithm starts from the beginning. This process is repeated until all bins are significantly different. 

# Parameters

None

# Methods

**fit**(self, x, y, alpha = 0.05, start_bins = 200, disp = True)

    Finds significant splits between bins, merges bins adjacent bins that don't have significantly different target 
    variable compositions
    
    Parameters: 
    
    x: array-like, of shape (n_samples, n_features)
      Contians the candidate continuous predictor variables to be binned
      
    y: array-like, of shape (n_samples, 1)
      An array contianing the binary target variable
    
    alpha: float
      The alpha level threshold to consider when finding significant differences between bins
      
    disp: bool, default = True
      Whether to display states updates about the binning of variables



**transform**(self, x)
    
    Transforms a test dataset accoring to the rule found when the fit function was was
    
    Parameters: 
    
    x: array-like, of shape (n_samples, n_features)
      Contians the candidate continuous predictor variables to be binned
      
    Returns: 
    
    variable_labels: array of shape (n_samples, n_features)
      Contains the binned continuous variable

**fit_transform**(self, x, y, alpha = 0.05, start_bins = 200, disp = True)

    Finds significant splits between bins, merges bins adjacent bins that don't have significantly different target 
    variable compositions
    
    Parameters: 
    
    x: array-like, of shape (n_samples, n_features)
      Contians the candidate continuous predictor variables to be binned
      
    y: array-like, of shape (n_samples, 1)
      An array contianing the binary target variable
    
    alpha: float
      The alpha level threshold to consider when finding significant differences between bins
      
    disp: bool, default = True
      Whether to display states updates about the binning of variables
      
      
    Returns: 
    
    variable_labels: array of shape (n_samples, n_features)
      Contains the binned continuous variable

# Attributes

**binning_rules**: array of shape (3, n_rules)
    
    Contains the rules learned on training data to bin the variables
