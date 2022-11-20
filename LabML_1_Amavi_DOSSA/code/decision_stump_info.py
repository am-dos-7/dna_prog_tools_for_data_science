import numpy as np
import utils
from decision_stump_error import DecisionStumpErrorRate

# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?

class DecisionStumpInfoGain(DecisionStumpErrorRate):
    def fit(self, X, y):
        # Overriding the fit() method
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self._splitSat = y_mode

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # self._minError = np.sum(y != y_mode)/N
        # entropy of all the set
        self._minError = entropy(np.bincount(y)/N)  

        # Loop over features looking for the best split

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Split the set according to value
                sat_set = y[X[:,d] >= value]
                not_set = y[X[:,d] < value]

                # the two variables below are reintroduced just for compatibility with the inherithance
                y_sat = utils.mode(sat_set)
                y_not = utils.mode(not_set)

                # Compute the probabilities of each set of elements in the two splits 
                # sat set
                p_sat = np.bincount(sat_set)/N
                e_sat = entropy(p_sat)

                # not-sat set
                p_not = np.bincount(not_set)/N
                e_not = entropy(p_not)

                # Compute error (weighted sum of the entropies of the two groups)
                errors = e_sat*np.size(sat_set)/N + e_not*np.size(not_set)/N

                # Compare to minimum error so far
                if errors < self._minError:
                    # This is the lowest error, store this value
                    self._minError = errors
                    self._splitVariable = d
                    self._splitValue = value
                    self._splitSat = y_sat
                    self._splitNot = y_not
    


    
"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
