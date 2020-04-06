class SupervisedBinning: 
    
    def __init__(self): 
        
        self.binning_rules = None
        
        
    
    def create_index(self, x, y):
        """
        Creates an index and binds it to a numpy array of data

        Parameters
        ----------
        x: array of shape (n_samples, n_features)
            Contains the array of variables to be binned against the target

        y: array of shape (n_samples, 1)
            Contians the binary target variable

        Returns
        -------
        bind: array (n_samples, n_features+2)
            An array cotaining (index, features, target)

        """
    
        x_dims = x.shape

        if len(x_dims) > 1:
            n_rows, n_cols = x_dims

        else:
            n_rows, n_cols = x_dims[0], 1

        index = np.array([i for i in range( n_rows )])

        bind = np.concatenate((index, x, y), axis = None)

        return bind.reshape(n_cols + 2, n_rows).T



    def sort_array(self, array, sort_by): 
        """Sorts a numpy array in ascending order

        Parameters
        ----------
        array: array to be sorted
            Contains the array of variables to be binned against the target

        sort_by: int
            Column index to sort by

        Returns
        -------
        array: array 
            The array sorted
        """
        return array[array[:,sort_by].argsort()]



    def equal_sample_bins(self, raw_values, num_bins):
        """Creates an array of bin labels with equal sample size on a sorted array of values

        Parameters
        ----------
        array: array to be sorted
            Contains the array of variables to create bin lables for

        num_bins: int
            The number of bin labels to create

        Returns
        -------
        bin_labels: array 
            An array of bin lables
        """

        bin_labels = pd.qcut(raw_values, num_bins, labels = False)

        return bin_labels



    def bin_ranges(self, raw_values, labels):

        """Calculates the the range of values that each bin contains

        Parameters
        ----------
        raw_values: array of shape (n_samples, 1)
            Contains a sorted array of raw continuous values corresponding to the bin labels

        labels: array of shape (n_samples, 1)
            Contains the bin labels corresponding to the raw values

        Returns
        -------
        array: array of shape (n_samples, 2)
            An array of [lower_bound, upper_bound] values 
        """

        obs = len(raw_values)

        lower_bounds = [None]*obs

        upper_bounds = [None]*obs

        unique_labels = np.unique(labels)

        start_index = 0

        for label in unique_labels: 

            subset = raw_values[labels == label]

            bin_size = len(subset)

            min_obs = min(subset)

            max_obs = max(subset)

            lower_bounds[start_index:start_index+bin_size] = [min_obs]*bin_size

            upper_bounds[start_index:start_index+bin_size] = [max_obs]*bin_size

            start_index += bin_size

        range_labels = np.array([lower_bounds, upper_bounds]).T

        return range_labels



    def merge_bins(self, labels, y, alpha = 0.01):
        """Merges the original bins down into bins with significant differences between them with respect to the
           target variable

        Parameters
        ----------        
        labels: array of shape (n_samples, 1)
            Contains the bin labels

        y: array of shape (n_samples, 1)
            Binary target variables 

        alpha: float, optional
            The amount of significnat difference between bins that must be present to prevent merging


        Returns
        ----------
        combined_labels: array of shape (n_samples, 1)
            Contains the new merged bins labels   
        """

        combined_labels = np.array(labels, copy = True)

        num_labels = len( np.unique(combined_labels) )

        stop_index = num_labels - 1

        stop = False

        index = 0

        while stop == False:

            unique_labels = np.unique(combined_labels)

            label1 = unique_labels[index]

            label2 = unique_labels[index+1]

            ct = self.crosstabulation(labels = combined_labels, 
                                 test_labels = [label1,label2], 
                                 y = y)

            pval = self.fisher_test(crosstable = ct)

            # If no significant difference exists we combine the groups
            if pval > alpha: 

                #print("Not Significant:", label1, label2, pval)
                # Change to index of both groups to the lower index value
                combined_labels[(combined_labels == label1) | (combined_labels == (label2))] = label1

                stop_index -= 1

                # We slide back one label to make sure the combining of two groups didn't affect
                # sigifincant between the combined group and the previous group
                if index == 0: 
                    # If we were exploring the first and second labels we continue as normal
                    None
                else: 
                    # If we are anywhere else we slide back one label group
                    index -= 1

                # If the index + 1 value is out of range then there are no more groups to check
                if index + 1 > stop_index: 
                    stop = True

            else: 
                #print("Significant:", label1, label2, pval)
                # If no group merge was made, we progress to the next pair investigation
                index += 1

                if index + 1 > stop_index: 
                    stop = True         

        return combined_labels




    def adjust_bins(self, labels): 
        """Takes the merged labels and adjusts then to have continuous bin labes (i.e. [1,5,10] becomes [0,1,2])

        Parameters
        ----------        
        labels: array of shape (n_samples, 1)
            Contains the merged bin labels from from the merge_bins(...) function


        Returns
        ----------
        combined_labels: array of shape (n_samples, 1)
            Contains the adjusted merged bins labels 
        """

        adjusted_labels = np.array( labels, copy = True)

        unique_labels = np.unique(labels)

        num_labels = len(unique_labels)

        new_labs = [i for i in range( len(unique_labels) )]

        for label in range(num_labels):

            old_label = unique_labels[label]

            new_label = new_labs[label]

            adjusted_labels[adjusted_labels == old_label] = new_label

        return adjusted_labels



    def create_rules(self, binned_ranges, labels): 
        """Creates binning rules for the test dataset based on binned value ranges

        Parameters
        ----------
        binned_ranges: array of shape (n_samples, 2)
            Contains the values ranges for each bin for each observation in the dataset

        labels: array of shape (n_samples, 1)
            Contains the bin labels 

        Returns
        ----------
        rules: array
            Contains the upper and lower bounds of values for each unique bin label in the dataset
        """

        unique_labels = np.unique(labels)

        lower = np.unique(binned_ranges[:,0])

        upper = np.unique(binned_ranges[:,1])

        rules = np.stack([unique_labels, lower, upper], axis = 1)

        return rules


    def crosstabulation(self, labels, test_labels, y): 
        """Create a cross table of value frequecies

        Parameters
        ----------        
        labels: array of shape (n_samples, 1)
            Contains the bin labels

        test_labels: list of length 2
            Contains the values of the two bins to test in the crosstabulation

        y: array of shape (n_samples, 1)
            Binary target variables 

        Returns
        ----------
        cross_table: array
            Contains the crosstabulations as (bins, targets)
        """

        bins_to_test = labels[(labels == test_labels[0]) | (labels==test_labels[1])]

        targets = y[(labels == test_labels[0]) | (labels==test_labels[1])]

        cross_table = pd.crosstab(bins_to_test, targets).to_numpy()

        return cross_table



    def fisher_test(self, crosstable):
        """Computes the p-value of fishers exact test for a 2x2 crosstabulation table

        Parameters
        ----------
        crosstable: array of shape (2x2)

        Returns
        ----------
        pvalue: float
            The pvalue of Fishers exact test

        """

        oddsratio, pvalue = stats.fisher_exact( crosstable )

        return pvalue

    
    
    def apply_rules(self, raw_values, rules): 
        """Applys the learned rules to an array of values


        Parameters
        ----------
        raw_values: array of shape (n_samples, 1)
            Contains an array of raw values to apply binning rules to

        rules: array of shape (n_rules, 3)
            Contains the rule index, lower bound, and upper bound for each rule

        Returns
        ----------
        test_labels: array of shape (n_samples, 1)
            The labeled values of the raw_values array 
        """
        test_labels = np.array(raw_values, copy = True)

        num_rules = len(rules)

        for rule in range(num_rules):

            # The first rule is value <= upper bound of first rule
            if rule == 0: 

                rule_structure = rules[rule]

                test_labels[test_labels <= rule_structure[-1]] = 0

            # The last rule if value > upper bound of previous rule
            elif rule == num_rules-1:

                previous_rule_structure = rules[rule-1]

                test_labels[test_labels > previous_rule_structure[-1]] = rule

            # All intermediate rules have (value > upper bound of previous rule) and (value <= upper bound of current rule)
            else:

                rule_structure = rules[rule]

                previous_rule_structure = rules[rule-1]

                test_labels[(test_labels > previous_rule_structure[-1]) & (test_labels <= rule_structure[-1])] = rule

        return test_labels
    
    
    
    def fit(self, x, y, alpha = 0.05, start_bins = 200, disp = True): 
        
        x = np.array(x)
        
        y = np.array(y)
        
        # Create an index on the data so that observations can be returned in their original order
        indexed_array = self.create_index(x = x, y = y)
        
        self.columns = len(x[0])
        
        self.binning_rules = [None]*self.columns
        
        for column in range(1, self.columns+1): 

            # Sort the array in ascending order by a given column
            sorted_array = self.sort_array(array = indexed_array, sort_by = column)

            # Create bin labels
            binned_labels = self.equal_sample_bins(raw_values = sorted_array[:, column], num_bins = start_bins)

            # Merge bin labels according to significance
            merged_bin_labels = self.merge_bins(labels = binned_labels, y = sorted_array[:, -1], alpha = alpha)
            
            # If all bins are merged into 1 then no significant splits were detected and we don't create a rule_set
            # and we move on to the next variable
            if len(merged_bin_labels) == 1: 
                if disp: 
                    print('No significant splits detected on variable', column)
                continue
            
            if disp: 
                print('Significant splits detected on variable', column)
                    
            # Adjust bin labels to have sequential labels
            merged_bin_labels = self.adjust_bins(merged_bin_labels)

            # Get the values boundries on each bin
            bin_range_values = self.bin_ranges(raw_values = sorted_array[:, column], labels = merged_bin_labels)

            # Create bin rules 
            rule_set = self.create_rules(binned_ranges = bin_range_values, labels = merged_bin_labels)
            
            self.binning_rules[column-1] = rule_set
    
    
    
    def transform(self, x):
        
        variable_labels = [None]*self.columns
        
        for i in range( self.columns ):
            
            rule_set = self.binning_rules[i]
            
            if type(rule_set) != type(None):
            
                variable_values = x[:, i]

                label_column = self.apply_rules(raw_values = variable_values, rules = rule_set)

                variable_labels[i] = label_column
                
        return np.array(variable_labels).T
    
    
    def fit_transform(self, x, y, alpha = 0.05, start_bins = 200, disp = True):
    
        self.fit(x, y, alpha = 0.05, start_bins = 200, disp = True)
        
        return self.transform(x)
        