'''
Created on Apr 21, 2017

@author: guilhermevarela
'''
import unittest
import pandas as pd 
from regress import lin
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
  
class TestRegress(unittest.TestCase):

    
    def testLinearRegression1D(self):
        
        df = pd.read_csv('../datasets/1d.csv', header=None)
        X = df.loc[:,0]
        Y = df.loc[:,1]

        
        
        W, rsq = lin(X,Y)    
        
        self.assertAlmostEqual(W[0], 2.8644240756607786, None, "W[0]*X + W[1] .: W[0] should ~ 2.8644", 0.0001)
        self.assertAlmostEqual(W[1], 1.972612167484588, None,  "W[0]*X + W[1] .: W[1] should ~ 1.9726", 0.0001)
        self.assertAlmostEqual(rsq, 0.99118382029778052, None,  "r2 ~ 0.99118", 0.0001)    
     
    def testLinearRegression2D(self):

        df = pd.read_csv('../datasets/2d.csv', header=None)
        X = df.loc[:,[0,1]]
        Y = df.loc[:,2]
        W, rsq = lin(X,Y, fill=True)    
        
        self.assertAlmostEqual(W[0], 2.01666793, None, "W[0] .: expected ~2.0166\ngot "  + str(W[0]), 0.0001)
        self.assertAlmostEqual(W[1], 2.96985048, None, "W[1] .: expected ~2.96985\ngot " + str(W[1]), 0.0001)
        self.assertAlmostEqual(W[2], 1.46191241, None, "W[2] .: expected ~1.46191\ngot " + str(W[2]), 0.0001)
        self.assertAlmostEqual(rsq, 0.99800406124757779, None,  "r2 .:   ~0.99800\ngot " + str(rsq) , 0.0001)
                

