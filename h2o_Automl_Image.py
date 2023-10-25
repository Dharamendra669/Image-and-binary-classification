#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2
import os
path = 'C:/Users/Asus/Desktop/Predictive Models/Django/model/'
source = 'C:/Users/Asus/Desktop/Predictive Models/Django/model/'
destination = 'C:/Users/Asus/Desktop/Predictive Models/Django/modelbackup/'
#Image processisng start from here
def image_proc(MYDIR):
    output = []
    images = []
    file_name=[]
    IMAGE_SIZE=(8,8)
    for file in tqdm(os.listdir(MYDIR)):
        try:
            img_path = os.path.join(MYDIR, file)
            # Open and resize the img
            #global image
            image = mpimg.imread(img_path)
                #image = image.resize(IMAGE_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
                #print(image)                # Append the image and its corresponding label to the output
            images.append((image))
                #labels.append(folder)
            file_name.append(file)
        except:
            print('Image not found')


    images = np.array(images, dtype = 'float32')
            #labels = np.array(labels, dtype = 'int32')  
            #print(images.shape)
    output.append((images, file_name))
    images_2d=images.reshape(images.shape[0],IMAGE_SIZE[0]*IMAGE_SIZE[1]*3) 
    dataset_df=pd.DataFrame(images_2d)
    output_df=pd.DataFrame(file_name,columns=['file_name'])
    output_df['Actual'] = output_df['file_name'].str.split('-').str[0]
    s = pd.concat([dataset_df ,output_df], axis=1, join='inner')
    s = s.sample(frac=1).reset_index(drop=True)
    s["Actual"]=s["Actual"].str.lower() 
    s['Actual'] = s['Actual'].astype('category')
    return s
#Image processing completed here


#training start frm here
#train_directory=r"/c01/home/lid4l0u/test_and_train/train"
#import h2o
#h2o.init()
#from h2o.automl import H2OAutoML
def h2otraining(train_directory):
    import h2o
    h2o.init()
    from h2o.automl import H2OAutoML
    h2otraining.input_dataframe=image_proc(train_directory)
    train = h2o.H2OFrame(h2otraining.input_dataframe)
    h2otraining.train_f=train.drop(train.ncol-2)
    h2otraining.automl = H2OAutoML(max_models = 5, max_runtime_secs=1000, seed = 1)
    h2otraining.automl.train(x=h2otraining.train_f.columns, y='Actual', training_frame=h2otraining.train_f)
    leader = h2otraining.automl.leaderboard
    print("Model Created.....")
    #model_path1=h2o.save_model(model=h2otraining.automl.leader,path=path,force=True)
    #model_path=model_path1
    #print('Model Saved to:' + model_path)
    #leader.head(rows=leader.nrows)
        
        
    #return 'Y'
    
        #h2o.init()
        
    #except:
       # return 'N'
#training ends here
#dir=r"/c01/home/lid4l0u/test_and_train/test"
#testing start here
def h20predict(test_directory):
    import h2o
    print('Prediction using h2o saved model')
        #for root, dirs, files in os.walk(path, topdown=False):
            #for name in files:
            #print(os.path.join(root, name))
            #data = h2o.import_file(os.path.join(root, name))
                #imported_model = h2o.load_model(os.path.join(root, name))
    #imported_model = h2o.load_model(r'C:/Users/Asus/Desktop/Predictive Models/Django/model/')
    input_dataframe=image_proc(test_directory)
    test = h2o.H2OFrame(input_dataframe)
    test_f=test.drop(test.ncol-2)
    print('Predictions started....')
    preds = h2otraining.automl.predict(test_f)
    final = test_f['Actual'].cbind(preds)
    #final_l=test['FORMAT'].cbind(final)
    final_f=test['file_name'].cbind(final)
    df = final_f.as_data_frame(use_pandas=True)
    h20predict.df_1=df.iloc[:,0:3]
    h20predict.df_1.to_csv(r'C:\Users\Asus\Desktop\Predictive Models\Django\result\H2O_predictions_result.csv')
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    h20predict.acc=(accuracy_score(h20predict.df_1['Actual'], h20predict.df_1['predict']))*100
    h20predict.n=confusion_matrix(h20predict.df_1['Actual'], h20predict.df_1['predict'])
    h20predict.df_1['Actual']=h20predict.df_1['Actual'].astype('category')
    h20predict.cm_df = pd.DataFrame(h20predict.n,index = h20predict.df_1['Actual'].cat.categories, columns = h20predict.df_1['Actual'].cat.categories)
    #h2o.export_file(h20predict.df_1,"'C:/Users/Asus/Desktop/Predictive Models/Django/result/H2O_predictions_result.csv",force = True)
    print('Prediction result saved to:/Django/result/H2O_predictions_result.csv')
    return h20predict.acc
    
       
#     except:
#         return 'Error in Prediction .please contact application support!'
#training start frm here
#train_directory=r"/c01/home/lid4l0u/test_and_train/train"

