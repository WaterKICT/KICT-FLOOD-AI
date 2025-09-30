
from io import StringIO
import sys
import time
from random import * # random 모듈에서 모든 것들을 가져다 쓰겠다는 의미
import ast
import os
from pickle import dump
from datetime import datetime, timedelta

def auto_cast(value: str):
    try:
        # parsing 
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # original
        return value

def main():
    
    datafilepath = sys.argv[1]
    modelParameterPath =  sys.argv[2]
    predRainPath = sys.argv[3]
    saveReportPath = sys.argv[4]
    modelName = sys.argv[5]    
    get_pypath = sys.argv[6]    

    import subprocess

    # check_pakage
    def ensure_package(pkg_name, import_name=None):
        import_name = import_name or pkg_name
        try:
            __import__(import_name)
        except ImportError:
            try:
                print(f"{pkg_name} is not installed. Installing now...")
                subprocess.run([get_pypath, "-m", "pip", "install", "--upgrade", "pip"], check=True)
                subprocess.run([get_pypath, "-m", "pip", "install", pkg_name], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[Module installation error] {pkg_name} Installation failed : {e}")

    # 필요한 라이브러리 확인 및 설치
    ensure_package("tensorflow", "tensorflow")
    ensure_package("scikit-learn", "sklearn")
    ensure_package("numpy", "numpy")
    ensure_package("matplotlib", "matplotlib")
    ensure_package("pandas", "pandas")
    
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    # Import library
    from sklearn.preprocessing import MinMaxScaler
    
    import matplotlib.pyplot as plt
    from matplotlib import gridspec    
    import matplotlib.ticker as ticker  

    # get_modelparameter

    # load modelParam
    modeldict_data = {}
    try:
        with open(modelParameterPath, "r", encoding="cp949") as file:
            for line in file:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "watershed_with_target":
                        modeldict_data[key] = value
                    else:
                        modeldict_data[key] = auto_cast(value)
    except FileNotFoundError:
        print(f"{modelParameterPath} File not found : ")
        sys.exit(1)
    except Exception as e:
        print(f"{modelParameterPath} Error occurred while processing the file : {e}")
        sys.exit(1)

    dataModelpath = modeldict_data.get("save_path")
    
    # run_predict    

    seq_length = modeldict_data.get("seq_length")
    drop_out_rate = modeldict_data.get("drop_out_rate")

    # 유역-예측강우정보 확인
    watershed_with_target = modeldict_data.get("watershed_with_target")

    # 예측강우 사용시, 예측강우 시간정보 확인  
    bPredictRain = modeldict_data.get("apply_realtime_Pred_WS")
    Predict_WS_hour = 0
    if (bPredictRain):
        Predict_WS_hour = modeldict_data.get("realtime_Pred_WS_time")
    
    # training parameter
    dataset_filename = datafilepath
    model_name = modelName
    save_path = saveReportPath #이미 "/" 포함됨   model_name+"/"
    drop_out_rate = modeldict_data.get("drop_out_rate")
    seq_length = modeldict_data.get("seq_length")
    hidden_dim = modeldict_data.get("hidden_dim") 
    size_of_batch = modeldict_data.get("size_of_batch")   # 한번에 학습시킬 자료의 단위
    learning_rate = modeldict_data.get("learning_rate") 
    iterations = modeldict_data.get("iterations")
    patience_num = modeldict_data.get("patience_num")
    training_rate = modeldict_data.get("training_rate")
    test_start = modeldict_data.get("test_start")
    target_point = dataset_filename[dataset_filename.find('Target_')+7:dataset_filename.find('Target_')+14]  # 데이터셋 파일이름에서 타겟 관측소 코드만 슬라이싱  기본적으로 'Target_'위치를 찾아서 상대적 위치 찾음
    watershed_with_target = modeldict_data.get("watershed_with_target")
    criteria_header = modeldict_data.get("criteria_header")
    criteria_wl = modeldict_data.get("criteria_wl")
    apply_realtime_Pred_WS = modeldict_data.get("apply_realtime_Pred_WS")#False
    realtime_Pred_WS_time = 0
    if (apply_realtime_Pred_WS):
        realtime_Pred_WS_time =  modeldict_data.get("realtime_Pred_WS_time")

    # data_column of X and Y in training dataset
    Data_X_Column = modeldict_data.get("Data_X_Column")  # X variable
    Data_Y_Column = modeldict_data.get("Data_Y_Column")

    data_dim = len(Data_X_Column)
    output_dim = len(Data_Y_Column)

    lead_time = "timeseries"    # 0.5h, 1h, 2h, 3h, 4h, 5h, 6h , timeseries

    Predict_time_leadtime = modeldict_data.get("Predict_time_leadtime")
    Predict_time_index = modeldict_data.get("Predict_time_index")

    # 데이터셋 파일 정보확인
    #data = pd.read_csv(dataset_filename , parse_dates=['Date'])
    data = pd.DataFrame()
    try:
        data = pd.read_csv(dataset_filename , parse_dates=['Date'])
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    except FileNotFoundError:
        print(f"[Prediction and Performance Analysis] Data file not found : {dataset_filename}")        
        sys.exit(1)
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to load data file : {e}")        
        sys.exit(1)

    # 데이터셋 헤더정보 확인
    header = data.columns.values

    # Y data를 수위 값이 아닌 수위변동 값으로 계산
    target_header = data.columns[1]
    Y_header = data.columns[Data_Y_Column]

    for i in range(len(Y_header)) :
        data[Y_header[i]] = eval('data.{}-data.{}'.format(Y_header[i],target_header))

    # rate of training data size = training data size / total data size
    train_size = int(len(data)*training_rate) 
    test_size = len(data)-int(len(data)*test_start)

    # make data of training and test data
    data_training = data[:train_size]
    data_test = data[int(len(data)*test_start):]

    # check the length of data
    print('Total = ',len(data))
    print('Test = ',len(data_test))

    # data slicing

    X_data_training_drop = data_training.iloc[:,Data_X_Column]
    Y_data_training_drop = data_training.iloc[:,Data_Y_Column]

    X_data_test_drop = data_test.iloc[:,Data_X_Column]
    Y_data_test_drop = data_test.iloc[:,Data_Y_Column]

    # extraction the header of X_data
    X_data_training_column = pd.DataFrame({"header" : X_data_training_drop.columns})
    X_data_training_column_list = list(X_data_training_column["header"])

    # extraction the header of XYdata
    Y_data_training_column = pd.DataFrame({"header" : Y_data_training_drop.columns})
    Y_data_training_column_list = list(Y_data_training_column["header"])

    # data normalize of training data    
    from pickle import load
    #X_scaler = load(open(dataModelpath + '/X_scaler.pkl', 'rb'))
    #Y_scaler = load(open(dataModelpath + '/Y_scaler.pkl', 'rb'))
    x_scaler_path = os.path.join(dataModelpath, 'X_scaler.pkl')
    y_scaler_path = os.path.join(dataModelpath, 'Y_scaler.pkl')

    try:
        with open(x_scaler_path, 'rb') as f:
            X_scaler = load(f)

        with open(y_scaler_path, 'rb') as f:
            Y_scaler = load(f)

    except FileNotFoundError as fe:
        print(f"{fe}")        
        sys.exit(1)
    except Exception as e:
        print(f"{e}")        
        sys.exit(1)

    X_data_training_scale = X_scaler.fit_transform(X_data_training_drop)
    Y_data_training_scale = Y_scaler.fit_transform(Y_data_training_drop)

    # make the input data of training model   

    # Training process

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping    

    # test process
    X_df = X_data_test_drop
    Y_df = Y_data_test_drop

    # data normalize of test data
    X_inputs = X_scaler.transform(X_df)
    Y_inputs = Y_scaler.transform(Y_df)

    ## Find a Temporal discontinuity for test data
    DeltaT_test = []
    data_test = data_test.reset_index(drop=True)
    for i in range(len(data_test)) :
        if i ==0 :
            temp_deltaT = 0
        else :
            temp_deltaT = int(((data_test['Date'][i]-data_test['Date'][i-1]).seconds)/60)
        DeltaT_test.append(temp_deltaT) 

    df_DeltaT_test = pd.DataFrame({'DeltaT_test':DeltaT_test})

    # make a check list for finding a Temporal discontinuity for test data
    check_test = []
    for n in range(len(data_test)):
        if n < (seq_length-1) :
            temp_check = ""
        else :
            temp_check = sum(df_DeltaT_test['DeltaT_test'][n+2-seq_length:n+1])
            if temp_check == 10*(seq_length-1) :
                temp_check = n
            else :
                temp_check = ""
                
        check_test.append(temp_check)

    df_check_test = pd.DataFrame({'Check_test':check_test})    

    X_test = []
    Y_test = []
    X_test_time = []
    X_test_data = []
    X_test_rainfall = []

    # make dataset of test
    for k in check_test:
        if k == "" :
            k
            
        else :
            X_test.append(X_inputs[k+1-seq_length:k+1,:])
            Y_test.append(Y_inputs[k,:])
            X_test_time.append(data_test['Date'][k])
            X_test_data.append(data_test[X_data_training_column_list[0]][k])
            X_test_rainfall.append(data_test['WS_{}'.format(watershed_with_target)][k]) 

    # make the result file
    Current_time_test = pd.DataFrame({'Date':X_test_time})
    Current_time_data = pd.DataFrame({X_data_training_column_list[0]:X_test_data})
    Current_time_watershed_rainfall = pd.DataFrame({'Rainfall': X_test_rainfall})

    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Apply Realtime Predict_WS 사용시,        
    if (bPredictRain == True):

        set_predPath = predRainPath
        predict_data_path = set_predPath # 예측 표준유역 강우데이터 경로
        
        post_list = X_data_training_column_list   

        #Predict WS 열에 대해서 예측결과를 불러와 기존 예측자료와 대체
        Predict_WS_column = []          
        for n in range(len(post_list)):
            if post_list[n].find('Predict_WS') >= 0 :
                exist_rain_file = os.path.join(predict_data_path,'{}.csv'.format(post_list[n][12:]))
                # 예측파일이 있을때,
                if os.path.isfile(exist_rain_file):                        
                    Predict_WS_column.append(n)
                    predict_data = pd.read_csv(os.path.join(predict_data_path,'{}.csv'.format(post_list[n][12:])), encoding='cp949', date_parser = True)  

                    # Convert to datetime for Date column
                    predict_data['Date'] = pd.to_datetime(predict_data['Date'])

                    # Current_time_test별 데이터 변경
                    for j in range(len(Current_time_test)):

                        # Current time 시간의 predict_data 인덱스 찾기
                        if (len(predict_data.index[predict_data['Date'] == str(Current_time_test['Date'][j])].to_list())>0 ):

                            Current_time_index = predict_data.index[predict_data['Date'] == str(Current_time_test['Date'][j])].to_list()
                            Currenttime = Current_time_index[0]
                            try:                            
                                for k in range(Predict_WS_hour*6) :
                                    X_test[j][seq_length-(Predict_WS_hour*6)+k][n] = (predict_data['RF{}'.format(k+1)][Currenttime]-X_scaler.data_min_[n])/X_scaler.data_range_[n]
                            except:
                                print ("warning")
                else:
                    print ("file not found - ", post_list[n][12:])

    # load the best model
    #new_model = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))
    try:
        new_model = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))
    except OSError:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} File not found : ")        
        sys.exit(1)
    except Exception as e:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} Failed to load model : {e}")        
        sys.exit(1)

    # calculate predict Y
    #Y_pred = new_model.predict(X_test)
    # conversion to original scale
    #Y_pred = Y_scaler.inverse_transform(Y_pred)
    #Y_test = Y_scaler.inverse_transform(Y_test)
    try:
        Y_pred = new_model.predict(X_test)
        Y_pred = Y_scaler.inverse_transform(Y_pred)
        Y_test = Y_scaler.inverse_transform(Y_test)
    except Exception as e:
        print(f"{e}")        
        sys.exit(1)

    # Make a prediction time for each leadtime
    Predict_time = pd.DataFrame()

    for i in range(0, len(Predict_time_leadtime)):
        Predict_time[Predict_time_index[i]] = pd.DatetimeIndex(Current_time_test['Date']) + timedelta(minutes=Predict_time_leadtime[i])
        
    result_all = pd.DataFrame()
    result_all = pd.concat([Current_time_test,Current_time_data,Current_time_watershed_rainfall],axis=1)

    for i in range(0,output_dim):
        result_tmp = pd.DataFrame({"Y_true_"+Predict_time_index[i] : Y_test[:,i],"Y_pred_"+Predict_time_index[i] : Y_pred[:,i]})
        Predict_time_tmp = Predict_time.iloc[:,[i]]
        result_all = pd.concat([result_all,Predict_time_tmp,result_tmp],axis=1)

    # Y data를 수위변동 값에서 수위값으로 계산
    result_column = result_all.columns
    Y_header=[]

    for j in range(len(result_column)):
        if 'Y_' in result_column[j] :
            Y_header.append(result_column[j])

    for j in range(0,len(Y_header),2) :
        result_all[Y_header[j]] = eval('result_all.{}+result_all.{}'.format(Y_header[j],target_header))
        result_all[Y_header[j+1]] = eval('result_all.{}+result_all.{}'.format(Y_header[j+1],target_header))

    result_all.to_csv(saveReportPath+'predict_result_'+model_name+'_timeseries_drop-out_'+str(drop_out_rate)+"_seq_length_"+str(seq_length)+'.csv', index=False)

    # Evaluating process
    import math

    # Analysis of each Criteria waterlevel
    NSE_total = pd.DataFrame()

    for n in range(len(criteria_wl)+1) :
        Y_true_filter = pd.DataFrame()
        Y_pred_filter = pd.DataFrame()
        
        if n==0 :
            for i in range(len(Predict_time_leadtime)):
                tmp_data_filter = result_all[['Y_true_{}'.format(Predict_time_index[i]),'Y_pred_{}'.format(Predict_time_index[i])]]
                
                Y_true_filter = pd.concat([Y_true_filter,tmp_data_filter['Y_true_{}'.format(Predict_time_index[i])]],axis=1)
                Y_pred_filter = pd.concat([Y_pred_filter,tmp_data_filter['Y_pred_{}'.format(Predict_time_index[i])]],axis=1)

            Y_true_filter, Y_pred_filter = np.array(Y_true_filter), np.array(Y_pred_filter)

        else : 
            for i in range(len(Predict_time_leadtime)):
                tmp_data = result_all[['Y_true_{}'.format(Predict_time_index[i]),'Y_pred_{}'.format(Predict_time_index[i])]]
                tmp_data_filter = tmp_data.drop(tmp_data[tmp_data['Y_true_{}'.format(Predict_time_index[i])]<criteria_wl[n-1]].index).reset_index(drop=True)

                Y_true_filter = pd.concat([Y_true_filter,tmp_data_filter['Y_true_{}'.format(Predict_time_index[i])]],axis=1)
                Y_pred_filter = pd.concat([Y_pred_filter,tmp_data_filter['Y_pred_{}'.format(Predict_time_index[i])]],axis=1)

            Y_true_filter, Y_pred_filter = np.array(Y_true_filter), np.array(Y_pred_filter)

        # Calculate NSE
        
        # Define average function
        if len(Y_true_filter) == 0:
            obs_avg = None
        else : 
            obs_avg = sum(Y_true_filter, 0.0) / len(Y_true_filter)   # calculate average

        # Define NSE(nash-sutcliffe efficiency)function	

        e1 = list()
        e2 = list()

        for i in range(0,len(Y_true_filter)):
            e1.append((Y_true_filter[i]-Y_pred_filter[i])**2)
            e2.append((Y_true_filter[i]-obs_avg)**2)

        if len(e2) ==0 or 0 in sum(e1) :
            nse= None
            if n==0 : 
                print('**No data for calculation of NSE')
            else :
                print('**No data for calculation of NSE over {}'.format(criteria_header[n-1]))
        else : 
            sum_e1 = sum(e1)
            sum_e2 = sum(e2)
            nse = 1- sum_e1/sum_e2   # calculate NSE
            
            if n==0 :
                print('== NSE ==')
                print('Data length = {}'.format(len(Y_true_filter)))
                print('Average of Obs. data=',obs_avg)
                print('NSE=',nse)
                
            else : 
                print('== NSE over {} =='.format(criteria_header[n-1]))
                print('Data length = {}'.format(len(Y_true_filter)))
                print('Average of Obs. data=',obs_avg)
                print('NSE=',nse)

            # Visualising the results
            if n==0 :
                save_graph_path = saveReportPath+'/'
                
                # Unless save_path folder exist, make the save_path folder
                if not os.path.exists(save_graph_path):
                    os.makedirs(save_graph_path)

                for i in range(0, len(Predict_time_index)):
                    plt.figure(figsize=(14,5))
                    plt.plot(Y_true_filter[:,i], color = 'red', label = 'Observed Water Level')
                    plt.plot(Y_pred_filter[:,i], color = 'blue', label = 'Predicted Water Level')
                    plt.title('Water Level Prediction_lead time_{}(NSE ={})'.format(Predict_time_index[i],str(nse[i])))
                    plt.xlabel('Time')
                    plt.ylabel('Water Level')
                    plt.legend()
                    #plt.savefig(save_graph_path+'prediction_result_'+model_name+'_after_'+Predict_time_index[i]+'_drop-out_'+str(drop_out_rate)+"_seq_length_"+str(seq_length)+'.png')
                    try:
                        plt.savefig(save_graph_path+'prediction_result_'+model_name+'_after_'+Predict_time_index[i]+'_drop-out_'+str(drop_out_rate)+"_seq_length_"+str(seq_length)+'.png')
                    except Exception as e:
                        print(f"{e}")
                    plt.close()
                    
                NSE_total = pd.concat([NSE_total,pd.DataFrame({'All_data':nse})],axis=1)
                NSE_total.to_csv(saveReportPath+'/NSE_total_{}.csv'.format(target_point), index=False)
                
            else :
                save_graph_path = saveReportPath+'/over_'+criteria_header[n-1]+'/'
            
                # Unless save_path folder exist, make the save_path folder
                if not os.path.exists(save_graph_path):
                    os.makedirs(save_graph_path)

                for i in range(0, len(Predict_time_index)):
                    plt.figure(figsize=(14,5))
                    plt.plot(Y_true_filter[:,i], color = 'red', label = 'Observed Water Level')
                    plt.plot(Y_pred_filter[:,i], color = 'blue', label = 'Predicted Water Level')
                    plt.title('Water Level Prediction_lead time_{}(NSE over {}={})'.format(Predict_time_index[i],criteria_header[n-1],str(nse[i])))
                    plt.xlabel('Time')
                    plt.ylabel('Water Level')
                    plt.legend()
                    #plt.savefig(save_graph_path+'prediction_result_'+model_name+'_after_'+Predict_time_index[i]+'_drop-out_'+str(drop_out_rate)+"_seq_length_"+str(seq_length)+'.png')
                    try:
                        plt.savefig(save_graph_path+'prediction_result_'+model_name+'_after_'+Predict_time_index[i]+'_drop-out_'+str(drop_out_rate)+"_seq_length_"+str(seq_length)+'.png')
                    except Exception as e:
                        print(f"{e}")
                    plt.close()
                
                NSE_total = pd.concat([NSE_total,pd.DataFrame({criteria_header[n-1]:nse})],axis=1)
                NSE_total.to_csv(saveReportPath+'/NSE_total_{}.csv'.format(target_point), index=False)   
    
if __name__ == '__main__':
    main()
    

        