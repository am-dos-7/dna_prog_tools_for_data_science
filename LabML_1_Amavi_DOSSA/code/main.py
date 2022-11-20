# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              
import pandas as pd                             
import matplotlib.pyplot as plt                 
from scipy.optimize import approx_fprime        
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`


""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

import utils
from decision_stump import DecisionStumpEquality
from decision_stump_generic import DecisionStumpEqualityGeneric
from decision_stump_error import DecisionStumpErrorRate
from decision_stump_info import DecisionStumpInfoGain
from decision_tree import DecisionTree



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # YOUR ANSWER HERE
        print("The minimum depth of a binary tree with 64 leaf nodes is: 6")
        print("The minimum depth of binary tree with 64 nodes (includes leaves and all other nodes) is: 6.") 
    
    elif question == "1.2":
        # YOUR ANSWER HERE
        print("The running time of the function", "func1 ", "is: O(N).")
        print("The running time of the function", "func2 ", "is: O(N), if we don't consider the vectorization aspect of numpy.")
        print("The running time of the function", "func3 ", "is: O(1).")
        print("The running time of the function", "func4 ", "is: O(N²).")
    
    elif question == "2.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values

        # YOUR CODE HERE
       
    elif question == "2.2":
        
        # YOUR CODE HERE : modify HERE
        figure_dic = {'A':1,
                      'B':2,
                      'C':3,
                      'D':4,
                      'E':5,
                      'F':6}
        for label in "ABCDEF":
            print("Match the plot", label, "with the description number: ",figure_dic[label])
        
    elif question == "3.1":
        # 1: Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3: Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with Equality rule error: %.3f"
              % error)

        # 4: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_1_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # YOUR ANSWER HERE
        print("It makes sense to use an  equality-based splitting rule rather than the threshold-based splits when the features' modality is binary " )
    
    elif question == "3.2":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the generic decision stump
        model = DecisionStumpEqualityGeneric()
        # model.fit_predict(X, y)

        print("Decision Stump Generic rule error: %.3f" % model.score(X, y))

        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "3.3":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the inequality decision stump
        model = DecisionStumpErrorRate()

        print("Decision Stump with inequality rule error: %.3f" %model.score(X, y))

        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
               
    elif question == "3.4":
        # 1: Load citiesSmall dataset         
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2: Evaluate the inequality decision stump
        model = DecisionStumpInfoGain()

        print("Decision Stump with Info Gain error: %.3f" %model.score(X, y))

        # 3: Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    
    elif question == "3.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q3_5_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # Here YOUR CODE
        print("The code corresping to this model is in the python file simple_decision.py.")

    elif question == "3.6":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try
       
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="errorrate")
        
        
        t = time.time()
        my_tree_errors_infogain = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3_6_tree_errors.pdf")
        plt.savefig(fname)


    else:
        print("No code to run for question", question)