# Supervised Binning
A Python class for binning continuous variables in a way that the bins significantly predict a binary target variable



# Author
1. Andrew Francis



# Overview
Intial characteristic analysis is a binning method that bins continuous predictor variables into disctrete categroical bins. This is useful when implementing a logistic regression model, but have variables that are not linearly associated with the logit. The hope is that the tool will prove useful for individuals practicing financial analytics in Python. 

Initial characteristic analysis starts by breaking the continuos variable into a pre-specifed number of bins with equal sample sizes across bins. Fishers exact test is used to determine if a significant difference bewtween adjacent bins exists. If no significant exists then the bin contents are merged and the algorithm starts from the beginning. This process is repeated until all bins are significantly different.



# Parameters

None



# Methods

**fit**(self, x, y, alpha, start_bins, disp)

    Finds significant splits between bins, merges bins adjacent bins that don't have significantly different target 
    variable compositions
    
    Parameters: 
    
    x: array-like, of shape (n_samples, n_features)
      Contians the candidate continuous predictor variables to be binned
      
    y: array-like, of shape (n_samples, 1)
      An array contianing the binary target variable
    
    alpha: float, default = 0.05
      The alpha level threshold to consider when finding significant differences between bins
      
    start_bins: int, default = 200
        How many equal sample bins to start the analysis with. It is recommended that the number of initial bins be set 
        so that each bin may contain at least 50 samples. 
      
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



**fit_transform**(self, x, y, alpha, start_bins, disp)

    Finds significant splits between bins, merges bins adjacent bins that don't have significantly different target 
    variable compositions
    
    Parameters: 
    
    x: array-like, of shape (n_samples, n_features)
      Contians the candidate continuous predictor variables to be binned
      
    y: array-like, of shape (n_samples, 1)
      An array contianing the binary target variable
    
    alpha: float, default = 0.05
      The alpha level threshold to consider when finding significant differences between bins
      
    start_bins: int, default = 200
        How many equal sample bins to start the analysis with. It is recommended that the number of initial bins be set 
        so that each bin may contain at least 50 samples. 
      
    disp: bool, default = True
      Whether to display states updates about the binning of variables
      
      
    Returns: 
    
    variable_labels: array of shape (n_samples, n_features)
      Contains the binned continuous variable



# Attributes

**binning_rules**: list of length n_features
    
    Contains the rules learned on training data to bin the variables. Contains a rule array for each feature in the dataset.
    The rule arrays are given as [bin_label, lower_bound, upper_bound]
    
    
    
# Use Case Example

Define some data: 

    x_train: Data used to find binning rules
    y_train: Target variable values used to train the splits
    x_test: Data used to evalute the model
    y_test: Data used to evaluate the model 



Initialize the SupervisedBinning object class:

    sbinning = SupervisedBinning()



Find split rules:

    sbinning.fit(x_train, y_train, alpha = 0.05, start_bins = 200, disp = True)

            This will display information about the status of the binning procedure: 
                Significant splits detected on variable 1
                Significant splits detected on variable 2
                Significant splits detected on variable 3



Show on of the split rule arrays:

    sbinning.binning_rules[1]
    
    Output: 
    array([[ 0.0000000e+00, -8.2286610e+01, -6.3280290e+01],
           [ 1.0000000e+00, -6.3235650e+01, -6.1001400e+01],
           [ 2.0000000e+00, -6.0957786e+01,  2.4889000e+04],
           [ 3.0000000e+00,  2.4891000e+04,  2.6989000e+04],
           [ 4.0000000e+00,  2.6994000e+04,  1.7974390e+06]])
           
    
Find the bin labels on the test dataset: 

    sbinning.transform(x_test)
    
    Output: 
    array([[5., 2., 2.],
           [5., 4., 4.],
           [5., 4., 4.],
           ...,
           [5., 2., 2.],
           [5., 2., 0.],
           [5., 2., 2.]])
           
# Dependencies
1. numpy
2. pandas
3. scipy
