import h2o
import os
import time
import seaborn
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
import shutil

import sys

#args = sys.argv  # a list of the arguments provided (str)
#print("running trainmodel.py", args)
#a, b,c = str(args[1]), str(args[2]), str(args[3])
#print(a, b, a + b)

#h2o.init()

path = '/Django/model/'
source = 'Django/model/'
destination = 'Django/modelbackup/'


def H2omodelcr(data):
    try:
        allfiles = os.listdir(source)
  
        for f in allfiles:
            shutil.move(source + f, destination + f)
            
        x = data.columns
        #print(x)
        y = data[-1].columns
        #print(y)
        x.remove(y[0])

        data[y] = data[y].asfactor()

        #original_model = H2OGeneralizedLinearEstimator()
        #original_model.train(x = x, y = y[0], training_frame=data)
    
        aml = H2OAutoML(max_models = 10, max_runtime_secs=120, seed = 1)
        aml.train(x = x,y = y[0], training_frame = data)

        print('Model Created.....')
        model_path1 = h2o.save_model(model=aml.leader, path=path, force=True)
        model_path = model_path1
        #original_model.save_mojo(path)
        print('Model Saved to:'+model_path)
        return 'Y'
    except:
        return 'N'
          
def H2omodelpr(new_observations):
    try :
        #sys.stdout = open('/c01/home/lid2t9w/Django/H2O_Prediction_Metrics.txt', 'w')

        print('Prediction Using H2O Model....')
        #path = '/c01/home/lid2t9w/Django/H2Omodel.zip'
          
        #imported_model = h2o.import_mojo(path)
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
            #print(os.path.join(root, name))
            #data = h2o.import_file(os.path.join(root, name))
                imported_model = h2o.load_model(os.path.join(root, name))
          
        x = new_observations.columns
        #print(x)
        y = new_observations[-1].columns
        #print(y)
        x.remove(y[0])

        new_observations[y] = new_observations[y].asfactor()

        print('Predictions started....')
        predictions = imported_model.predict(new_observations)
          
        new_observations['MachinePrediction'] = predictions['predict']

        h2o.export_file(new_observations,"/Django/result/H2O_predictions_result.csv",force = True)
        print('Prediction result saved to:/Django/result/H2O_predictions_result.csv')
        auc = round(imported_model.auc("valid = true")*100)
        
        return auc
    except:
        return 'Error in Prediction. please contact application support!'
        
#data = h2o.import_file(a+b)
#H2omodelcr(data)
          
#new_observations = h2o.import_file(a+c)
#H2omodelpr(new_observations)      
          

    