import h2o
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2
import os
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,session
import json
from json import JSONEncoder
#from h2o.automl import H2OAutoML
import h2o_Automl_Image
#h2o.init()
UPLOAD_FOLDER='uploads'
UPLOAD_FOLDER_TRAIN = 'uploads/train'
UPLOAD_FOLDER_TEST = 'uploads/test'
RESULT_FOLDER = 'Django/result/'
#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_TRAIN
app.config['UPLOAD_FOLDER_TRAIN'] = UPLOAD_FOLDER_TRAIN
app.config['UPLOAD_FOLDER_TEST'] = UPLOAD_FOLDER_TEST
app.secret_key = 'a random string'
#dropdown page
# @app.route('/', methods=['GET','POST'])
# def dropdown():
#     dies = ['diebteics', 'covid']
#     return render_template('test.html', colours=dies)


# Upload API
# def drop_down_select():
#     if request.method == 'POST':
#         tvalue = request.form['colour']
#         if tvalue=='covid':
#             return redirect('/BuildModel')
#         else:
#             return redirect('/BuildModel')
@app.route('/', methods=['GET','POST'])
def reroot():
    return redirect('/BuildModel')
@app.route('/BuildModel', methods=['GET', 'POST'])
def upload_file_covid():
    if request.method == 'POST':

        # check if the post request has the file part
        ifiles=request.files.getlist("ifile[]")
        for ifile in ifiles:
            filename = secure_filename(ifile.filename)
            ifile.save(os.path.join(app.config['UPLOAD_FOLDER_TRAIN'], ifile.filename))
        nfiles=request.files.getlist("nfile[]")        
        for nfile in nfiles:
            filename = secure_filename(nfile.filename)

            nfile.save(os.path.join(app.config['UPLOAD_FOLDER_TEST'], nfile.filename))







            #Process the model H2O
#             ifile_path = UPLOAD_FOLDER + filename
#             nfile_path = UPLOAD_FOLDER + nfilename
#             data = h2o.import_file(ifile_path)
#             ndata = h2o.import_file(nfile_path)

#             rows1 = data.shape[0]
#             column1 = (data.shape[1])-1
#             nrow = ndata.shape[0]

#             df_pos = data[data['Outcome'] == 1]
#             df_neg = data[data['Outcome'] == 0]
#             pos = df_pos.shape[0]
#             neg = df_neg.shape[0]

            modelbuild = h2o_Automl_Image.h2otraining(UPLOAD_FOLDER_TRAIN)
            nrows_train=len(h2o_Automl_Image.h2otraining.train_f)
            #if modelbuild == 'Y':
            auc = h2o_Automl_Image.h20predict(UPLOAD_FOLDER_TEST)
            nrows_test=len(h2o_Automl_Image.h20predict.df_1)
            
            data=h2o_Automl_Image.h20predict.df_1
            heading=(data.columns).tolist()
            v=(data.values).tolist()
            json_str_heading = json.dumps(heading)
            json_str_v = json.dumps(v)
#             q = {'data':data}
#             Table = []
#             for key, value in q.items():    # or .items() in Python 3
#                 temp = []
#                 temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
#                 Table.append(temp)
            
            #else:
                #return 'Model not build!! please contact Application support'
            #here code for printing actual value and total predicted value from the test data set
            t=h2o_Automl_Image.h20predict.df_1['Actual'].cat.categories
            rows = len(h2o_Automl_Image.h20predict.n);  
            cols = len(h2o_Automl_Image.h20predict.n[0]);
            q={}
            for i in range(0, rows):  
                sumCol = 0;  
                for j in range(0, cols):  
                    sumCol = sumCol + h2o_Automl_Image.h20predict.n[j][i];  
                q[i]="Total machine predicted value of " + t[i]+" is " + str(sumCol)+ " and truely machine predicted value for "+t[i]+" is "+str(h2o_Automl_Image.h20predict.n[i][i]) 

            #result = 'H2O_predictions_result.csv'
            #result_dt = h2o.import_file(RESULT_FOLDER + 'H2O_predictions_result.csv')
            #df_npos = result_dt[result_dt['MachinePrediction'] == 1]
            #df_nneg = result_dt[result_dt['MachinePrediction'] == 0]
            #npos = df_npos.shape[0]
            #nneg = df_nneg.shape[0]
            result = {'value':'H2O_predictions_result.csv',
                     'rows_train':nrows_train,
                      'rows_test':nrows_test,
                      'cm_value':q,
                      'auc':auc,
                      'headings':heading,
                      'v':v
                      
                     }
#             result = {'value':'H2O_predictions_result.csv',
#                       'rows':rows1,
#                       'column':column1,
#                       'pos':pos,
#                       'neg':neg,
#                       'nrow':nrow,
#                       'npos':npos,
#                       'nneg':nneg,
#                       'auc':auc}

            session['result'] = result
            #send file name as parameter to downlad
            return redirect('/modelsummary/'+ result.get('value'))
            #return aml.explain(ndata) 
    return render_template('upload_file_covid.html')
#output
@app.route("/modelsummary/<result>", methods = ['GET'])
def download_file_covid(result):
#     try:
    dicta = session['result'] 
        
#     except IOError:
#         print( 'Image not found')
    return render_template('download_covid.html',dicta = dicta)


# @app.route("/modelsummary/<result>", methods=['GET'])
# def data_shown(result):
#     h=session[result['headings']]
#     d=sesssion[result['v']]
#     return render_templates('download_covid.html',headings=h,datas=d)
# #image printing in 
# @app.route("/image/<result>",methods = ['GET'])
# def return_image ():
    
#     text = request.args.get('text')
# #     from PIL import Image 
#     input_img=str(text)

#     im3 = os.path.join(app.config['UPLOAD_FOLDER_TEST'], 'input_img') 
#     print(im3)
#     print(im3)
#     #im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
# #     plt.imshow(im3)
# #     plt.show()
@app.route('/return-files/<result>')
def return_files_tut_covid(result):

    file_path = RESULT_FOLDER + result
    return send_file(file_path, as_attachment=True, attachment_filename='')
    return render_templates('download_covid.html',user_image=im3 )
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000, debug=False)