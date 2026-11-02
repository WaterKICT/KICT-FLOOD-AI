import sys
import time
from datetime import timedelta
from random import *
from pickle import dump
#import time
import ast
import os
import numpy as np

def auto_cast(value: str):
    try:
        # parsing 
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # original
        return value

def main():
   
    get_pypath = sys.argv[2]
    predict_data_path = sys.argv[3]

    import subprocess

    # 패키지 확인 및 설치 함수
    def ensure_package(pkg_name, import_name=None):
        import_name = import_name or pkg_name
        try:
            __import__(import_name)
        except ImportError:
            print(f"{pkg_name} is not installed. Installing now...")
            subprocess.run([get_pypath, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            subprocess.run([get_pypath, "-m", "pip", "install", pkg_name], check=True)

    # 필요한 라이브러리 확인 및 설치
    ensure_package("tensorflow", "tensorflow")
    ensure_package("scikit-learn", "sklearn")
    ensure_package("numpy", "numpy")
    ensure_package("matplotlib", "matplotlib")
    ensure_package("pandas", "pandas")
    
    # make_model (GRU)
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    get_parampath = sys.argv[1]
    model_parameter = get_parampath

    modeldict_data = {} 
    try:
        with open(model_parameter, "r", encoding="cp949") as file:
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
        print(f"{model_parameter} File not found : ")
        sys.exit(1)
    except Exception as e:
        print(f"{model_parameter} Error occurred while processing the file : {e}")
        sys.exit(1)

    # training parameter
    dataset_filename = modeldict_data.get("data_path") #nn
    model_name = modeldict_data.get("model_name")   #nn
    save_path = modeldict_data.get("save_path") + "/" #model_name+"/"
    drop_out_rate = modeldict_data.get("drop_out_rate")     #nn  #  0~1
    seq_length = modeldict_data.get("seq_length")   #nn
    hidden_dim = modeldict_data.get("hidden_dim")   #nn
    size_of_batch = modeldict_data.get("size_of_batch") #nn   # 한번에 학습시킬 자료의 단위
    learning_rate = modeldict_data.get("learning_rate") #nn
    iterations = modeldict_data.get("iterations")   #nn
    patience_num = modeldict_data.get("patience_num")   #nn
    training_rate = modeldict_data.get("training_rate") #nn
    test_start = modeldict_data.get("test_start")#nn
    random_number = modeldict_data.get("random_number")
    target_point = dataset_filename[dataset_filename.find('Target_')+7:dataset_filename.find('Target_')+14]  # 데이터셋 파일이름에서 타겟 관측소 코드만 슬라이싱  기본적으로 'Target_'위치를 찾아서 상대적 위치 찾음
    watershed_with_target = modeldict_data.get("watershed_with_target")#nn
    criteria_header = modeldict_data.get("criteria_header") #nn
    criteria_wl = modeldict_data.get("criteria_wl")#nn
    apply_realtime_Pred_WS = modeldict_data.get("apply_realtime_Pred_WS")#False
    realtime_Pred_WS_time = 0
    if (apply_realtime_Pred_WS):
        realtime_Pred_WS_time =  modeldict_data.get("realtime_Pred_WS_time")

    # data_column of X and Y in training dataset
    Data_X_Column = modeldict_data.get("Data_X_Column") #nn  # X variable
    Data_Y_Column = modeldict_data.get("Data_Y_Column") #list(range(nn,nn+36,1))

    data_dim = len(Data_X_Column)
    output_dim = len(Data_Y_Column)

    # prediction time(lead time)
    lead_time = "timeseries"    # 0.5h, 1h, 2h, 3h, 4h, 5h, 6h , timeseries

    Predict_time_leadtime = modeldict_data.get("Predict_time_leadtime")  #list(range(10,370,10))
    Predict_time_index = modeldict_data.get("Predict_time_index") #[]

    # hidden layer parameter
    validation_rate = modeldict_data.get("validation_rate")#nn
    hidden_layer_unit = modeldict_data.get("hidden_layer_unit")#nn
    hidden_layer = len(hidden_layer_unit)

    activation_func = modeldict_data.get("activation_func")#nn  # 학습 모델의 activation function
    optimize_func = modeldict_data.get("optimize_func")#nn   # 학습 모델의 optimize function
    loss_func = modeldict_data.get("loss_func")# nn  # 학습 모델의 loss function

    # save the start time
    start = time.time() 

    # make a random number
    random_number = int(random_number)

    # setting the random number with reproductive
    tf.random.set_seed(random_number)

    #####################################################################   

    # read input data, Convert to datetime for Date column
    data = pd.DataFrame()
    try:
        data = pd.read_csv(dataset_filename, parse_dates=['Date'])
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        #data = data.dropna(subset=['Date']).reset_index(drop=True)
    except FileNotFoundError:
        print(f"[Model Creation] Data file not found : {dataset_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[Model Creation] Failed to load data file : {e}")
        sys.exit(1)

    # rate of training data size = training data size / total data size
    train_size = int(len(data) * training_rate)
    test_size = len(data) - int(len(data) * test_start)

    # make data of training and test data
    data_training = data.iloc[:train_size].copy()
    data_test = data.iloc[int(len(data) * test_start):].copy()

    # check the length of data
    print('Total = ', len(data))
    print('Train = ', len(data_training))
    print('Test = ', len(data_test))

    # data slicing
    X_data_training_drop = data_training.iloc[:, Data_X_Column]
    Y_data_training_drop = data_training.iloc[:, Data_Y_Column]

    X_data_test_drop = data_test.iloc[:, Data_X_Column]
    Y_data_test_drop = data_test.iloc[:, Data_Y_Column]

    # extraction the header of X/Y data
    X_data_training_column_list = X_data_training_drop.columns.tolist()
    Y_data_training_column_list = Y_data_training_drop.columns.tolist()

    # data normalize of training data
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_data_training_scale = X_scaler.fit_transform(X_data_training_drop)
    Y_data_training_scale = Y_scaler.fit_transform(Y_data_training_drop)

    # Unless save_path folder exist, make the save_path folder
    os.makedirs(save_path, exist_ok=True)

    # save the scaler
    with open(os.path.join(save_path, 'X_scaler.pkl'), 'wb') as f:
        dump(X_scaler, f)
    with open(os.path.join(save_path, 'Y_scaler.pkl'), 'wb') as f:
        dump(Y_scaler, f)

    ## Find a Temporal discontinuity for training data    
    DeltaT = []

    data_training = data_training.reset_index(drop=True)    # add - initialize_index

    for i in range(len(data_training)):
        if i == 0:
            temp_deltaT = 0
        else:
            temp_deltaT = int((data_training['Date'].iloc[i] - data_training['Date'].iloc[i - 1]).total_seconds() / 60)
        DeltaT.append(temp_deltaT)

    df_DeltaT = pd.DataFrame({'DeltaT': DeltaT})

    # make a check list for finding a Temporal discontinuity for training data
    check = []
    for n in range(len(data_training)):
        if n < (seq_length-1) :
            temp_check = ""
        else :
            temp_check = sum(df_DeltaT['DeltaT'][n+2-seq_length:n+1])
            if temp_check == 10*(seq_length-1) :
                temp_check = n
            else :
                temp_check = ""
                
        check.append(temp_check)

    df_check = pd.DataFrame({'Check':check})    

    # make the input data of training model
    X_train = []
    Y_train = []

    for k in check:
        if k == "":
            continue
        else:
            X_train.append(X_data_training_scale[k + 1 - seq_length:k + 1, :])
            Y_train.append(Y_data_training_scale[k, :])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    if len(X_train) == 0 or len(Y_train) == 0:
        print("[Model Creation] No valid continuous training sequences were found.")
        sys.exit(1)

    ### LSTM -> GRU 변경
    # Training process

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, GRU, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    regressior = Sequential()

    # activate the Early stop and checkpoint 
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_num)
    mc = ModelCheckpoint(os.path.join(save_path, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True)

    # check
    class ConsoleLogger(tf.keras.callbacks.Callback):
        def __init__(self, log_file_path):
            super().__init__()
            self.log_file_path = log_file_path
            self.log_file = open(self.log_file_path, "w", encoding="utf-8")

        def on_epoch_end(self, epoch, logs=None):            
            logs = logs or {}
            msg = (
                f"Epoch {epoch+1} - "
                f"loss: {logs.get('loss', 0):.4f}, "
                f"val_loss: {logs.get('val_loss', 0):.4f}, "
                f"accuracy: {logs.get('accuracy', 0):.4f}, "
                f"val_accuracy: {logs.get('val_accuracy', 0):.4f}, "
                f"mae: {logs.get('mae', 0):.4f}, "
                f"val_mae: {logs.get('val_mae', 0):.4f}"
                "\n"
            )
            print(msg, flush=True)
            self.log_file.write(msg)
            self.log_file.flush()

        def on_train_end(self, logs=None):
            self.log_file.close()


    regressior.add(GRU(units = hidden_dim, activation = activation_func, return_sequences = True, input_shape = (X_train.shape[1], data_dim)))
    regressior.add(Dropout(drop_out_rate)) # prevent overfitting

    # add hidden layers 
    for i in range(0, hidden_layer):
        regressior.add(GRU(units = hidden_layer_unit[i], activation = activation_func, return_sequences = True))
        regressior.add(Dropout(drop_out_rate))  # prevent overfitting

    regressior.add(GRU(units = hidden_dim, activation = activation_func))
    regressior.add(Dropout(drop_out_rate))  # prevent overfitting
    regressior.add(Dense(units = output_dim))
    regressior.summary()
    regressior.compile(optimizer=optimize_func, loss = loss_func, metrics = ['acc'])

    # 추가 (model_log)
    logger = ConsoleLogger(os.path.join(save_path, "model_log.txt"))

    history = regressior.fit(X_train, Y_train, epochs=iterations, batch_size=size_of_batch, validation_split = validation_rate, callbacks=[es,mc, logger], verbose=1)

    history.history.keys()
    his_dict = history.history
    loss = his_dict['loss']
    val_loss = his_dict['val_loss']

    # display the elapsed time of training
    Elapsed_time = int(time.time()-start)
    print("Elapsed time :"+str(Elapsed_time)+"sec")

    # make the loss result file
    result = pd.DataFrame({"loss" : loss[:],"val_loss" : val_loss[:]})
    result.to_csv(
        os.path.join(save_path, f'predict_result_{lead_time}_drop-out_{drop_out_rate}_seq_length_{seq_length}_loss_data.csv'),
        index=False
    )

    epochs = range(1, len(loss) + 1 )
    fig = plt.figure(figsize = (10, 5))

    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
    ax1.plot(epochs, val_loss, color = 'orange', label = 'val_loss')
    ax1.set_title('train and val loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    acc = his_dict.get('acc', his_dict.get('accuracy', []))
    val_acc = his_dict.get('val_acc', his_dict.get('val_accuracy', []))

    if len(acc) == len(epochs) and len(val_acc) == len(epochs):
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, acc, color='blue', label='train_acc')
        ax2.plot(epochs, val_acc, color='orange', label='val_acc')
        ax2.set_title('train and val acc')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('acc')
        ax2.legend()

    
    plt.savefig(
        os.path.join(save_path, f'training_loss_acc_{lead_time}_drop-out_{drop_out_rate}_seq_length_{seq_length}.png')
    )
    plt.close()

    # model save
    regressior.save(
        os.path.join(save_path, f"{model_name}_after_{lead_time}_drop-out_{drop_out_rate}_seq_length_{seq_length}")
    )

    #####################################################################    

    # test process
    data_test = data_test.reset_index(drop=True)
    X_df = X_data_test_drop
    Y_df = Y_data_test_drop

    # 테스트 데이터가 없을 경우 모델 생성은 완료하고 평가만 건너뜀
    if len(data_test) == 0 or X_df.empty or Y_df.empty:
        print("[Model Creation] Model was created successfully. However, there was no test data, so prediction/evaluation was skipped.")
        try:
            with open(os.path.join(save_path, "no_test_data.flag"), "w", encoding="utf-8") as f:
                f.write("NO_TEST_DATA\n")
        except Exception as e:
            print(f"[Model Creation] Failed to create no_test_data.flag : {e}")
        sys.exit(0)

    # data normalize of test data
    try:
        X_inputs = X_scaler.transform(X_df)
        Y_inputs = Y_scaler.transform(Y_df)
    except Exception as e:
        print(f"[Model Creation] Failed to normalize test data : {e}")
        sys.exit(1)

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

    ws_col_name = 'WS_{}'.format(watershed_with_target)
    if ws_col_name not in data_test.columns:
        print(f"[Model Creation] Column not found : {ws_col_name}")
        sys.exit(1)

    # make dataset of test
    for k in check_test:
        if k == "" :
            k
            
        else :
            X_test.append(X_inputs[k+1-seq_length:k+1,:])
            Y_test.append(Y_inputs[k,:])
            X_test_time.append(data_test['Date'][k])
            X_test_data.append(data_test[X_data_training_column_list[0]][k])
            X_test_rainfall.append(data_test[ws_col_name].iloc[k])

    # make the result file
    Current_time_test = pd.DataFrame({'Date': X_test_time})
    Current_time_data = pd.DataFrame({X_data_training_column_list[0]: X_test_data})
    Current_time_watershed_rainfall = pd.DataFrame({'Rainfall': X_test_rainfall})

    X_test, Y_test = np.array(X_test, dtype=np.float32), np.array(Y_test, dtype=np.float32)    

    # Apply Realtime Predict_WS 사용할 경우, 유역예측강우를 실제강우로 대체하여 적용
    if apply_realtime_Pred_WS:
        post_list = X_data_training_column_list

        # Predict WS 열에 대해서 예측결과를 불러와 기존 예측자료와 대체
        Predict_WS_column = []
        for n in range(len(post_list)):
            if post_list[n].find('Predict_WS') >= 0:
                exist_rain_file = os.path.join(predict_data_path, '{}.csv'.format(post_list[n][12:]))
                # 예측파일이 있을때,
                if os.path.isfile(exist_rain_file):
                    Predict_WS_column.append(n)
                    try:
                        predict_data = pd.read_csv(exist_rain_file, encoding='cp949', parse_dates=['Date'])
                        predict_data['Date'] = pd.to_datetime(predict_data['Date'], errors='coerce')
                    except Exception as e:
                        print(f"[Model Creation] Failed to read predict rainfall file : {exist_rain_file}, {e}")
                        continue

                    # Current_time_test별 데이터 변경
                    for j in range(len(Current_time_test)):
                        # Current time 시간의 predict_data 인덱스 찾기
                        if (len(predict_data.index[predict_data['Date'] == Current_time_test['Date'].iloc[j]].to_list()) > 0):
                            Current_time_index = predict_data.index[predict_data['Date'] == Current_time_test['Date'].iloc[j]].to_list()
                            Currenttime = Current_time_index[0]

                            for k in range(int(realtime_Pred_WS_time) * 6):
                                rf_col = 'RF{}'.format(k+1)
                                if rf_col not in predict_data.columns:
                                    continue
                                scale_range = X_scaler.data_range_[n]
                                if scale_range == 0:
                                    X_test[j][seq_length-(int(realtime_Pred_WS_time)*6)+k][n] = 0.0
                                else:
                                    X_test[j][seq_length-(int(realtime_Pred_WS_time)*6)+k][n] = (predict_data[rf_col].iloc[Currenttime] - X_scaler.data_min_[n]) / scale_range
                else:
                    print("file not found - ", post_list[n][12:])

    #load the best model
    try:
        new_model = tf.keras.models.load_model(os.path.join(save_path, 'best_model.h5'))
    except OSError:
        print(f"{os.path.join(save_path, 'best_model.h5')} File not found : ")
        sys.exit(1)
    except Exception as e:
        print(f"{os.path.join(save_path, 'best_model.h5')} Failed to load model : {e}")
        sys.exit(1)


    #calculate predict Y    
    try:
        Y_pred = new_model.predict(X_test)
        # conversion to original scale
        Y_pred = Y_scaler.inverse_transform(Y_pred)
        Y_test = Y_scaler.inverse_transform(Y_test)
    except Exception as e:
        print(f"[Model Creation] Prediction failed : {e}")
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

    result_all.to_csv(os.path.join(save_path, f'predict_result_{model_name}_timeseries_drop-out_{drop_out_rate}_seq_length_{seq_length}.csv'), index=False)
    
    # Evaluating process
    import math

    # Define average function
    def average(values):
        if len(values) == 0:
            return None
        return sum(values, 0.0) / len(values)

    obs_avg = average(Y_test)  # calculate average

    # Define NSE(nash-sutcliffe efficiency)function	

    def nse(obs_values,sim_values):
        obs_values = np.array(obs_values)
        sim_values = np.array(sim_values)

        # 데이터가 없을 경우
        if len(obs_values) == 0:
            return None

        # 관측값 평균
        obs_avg = np.mean(obs_values, axis=0)

        # NSE 계산
        e1 = (obs_values - sim_values) ** 2
        e2 = (obs_values - obs_avg) ** 2

        sum_e1 = np.sum(e1, axis=0)
        sum_e2 = np.sum(e2, axis=0)

        # 분모가 0이면 NSE 계산 불가
        nse_result = np.where(sum_e2 == 0, None, 1 - (sum_e1 / sum_e2))

        return nse_result

    nse = nse(Y_test, Y_pred)  # calculate NSE

    # print evaluation factor
    print("Average of Obs. data=",obs_avg)
    print("NSE=",nse)
    # Visualising the results

    for i in range(0, len(Predict_time_index)):
        plt.figure(figsize=(14,5))
        plt.plot(Y_test[:,i], color = 'red', label = 'Observed Water Level')
        plt.plot(Y_pred[:,i], color = 'blue', label = 'Predicted Water Level')
        plt.title('Water Level Prediction_lead time_{}(NSE={})'.format(Predict_time_index[i],str(nse[i])))
        plt.xlabel('Time')
        plt.ylabel('Water Level')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'prediction_result_{model_name}_after_{Predict_time_index[i]}_drop-out_{drop_out_rate}_seq_length_{seq_length}.png'))
        plt.close()
        
    # Analysis of each Criteria waterlevel

    NSE_total = pd.DataFrame()
    NSE_total = pd.concat([NSE_total,pd.DataFrame({'All_data':nse})],axis=1)

    for n in range(len(criteria_wl)) :
        nse_filter = list()

        # 해당 기준수위 조건에 실제 자료가 하나라도 있는지 확인
        has_criteria_data = False

        # Visualising the results
        save_graph_path = os.path.join(save_path, f'over_{criteria_header[n]}')

        for i in range(len(Predict_time_leadtime)):
            tmp_data = result_all[
                [
                    'Y_true_{}'.format(Predict_time_index[i]),
                    'Y_pred_{}'.format(Predict_time_index[i])
                ]
            ]

            tmp_data_filter = tmp_data.drop(
                tmp_data[
                    tmp_data[
                        'Y_true_{}'.format(Predict_time_index[i])
                    ] < criteria_wl[n]
                ].index
            ).dropna().reset_index(drop=True)

            Y_true_filter = np.array(
                tmp_data_filter[
                    'Y_true_{}'.format(Predict_time_index[i])
                ],
                dtype=float
            )

            Y_pred_filter = np.array(
                tmp_data_filter[
                    'Y_pred_{}'.format(Predict_time_index[i])
                ],
                dtype=float
            )

            # Calculate NSE
            if len(Y_true_filter) == 0:
                nse_filter.append(np.nan)
                continue

            # 해당 기준수위에 자료가 하나라도 있음을 기록
            has_criteria_data = True
            #else:
            obs_avg = np.mean(Y_true_filter)

            sum_e1 = np.sum((Y_true_filter - Y_pred_filter) ** 2)
            sum_e2 = np.sum((Y_true_filter - obs_avg) ** 2)

            # sum_e2가 0이면 NSE 계산 불가 → nan 처리
            if np.isfinite(sum_e2) and sum_e2 != 0:
                nse_filter.append(1 - (sum_e1 / sum_e2))
            else:
                nse_filter.append(np.nan)

            # 실제 자료가 있을 때만 폴더 생성 및 그래프 저장
            if not os.path.exists(save_graph_path):
                os.makedirs(save_graph_path)

            plt.figure(figsize=(14,5))
            plt.plot(Y_true_filter, color = 'red', label = 'Observed Water Level')
            plt.plot(Y_pred_filter, color = 'blue', label = 'Predicted Water Level')
            plt.title('Water Level Prediction_lead time_{}(NSE over {}={})'.format(Predict_time_index[i],criteria_header[n],str(nse_filter[i])))
            plt.xlabel('Time')
            plt.ylabel('Water Level')
            plt.legend()
            plt.savefig(os.path.join(save_graph_path, f'prediction_result_{model_name}_after_{Predict_time_index[i]}_drop-out_{drop_out_rate}_seq_length_{seq_length}.png'))
            plt.close()

        # 기준수위 조건을 만족하는 데이터가 전혀 없는 경우
        if not has_criteria_data:
            print(
                '**No data for calculation of NSE over {}'.format(
                    criteria_header[n]
                )
            )
            continue

        # 자료가 있는 기준수위만 CMD 출력 및 NSE CSV 컬럼 생성
        print('== NSE over {} =='.format(criteria_header[n]))
        print('NSE=', nse_filter)

        NSE_total = pd.concat([NSE_total,pd.DataFrame({criteria_header[n]:nse_filter})],axis=1)
        NSE_total.to_csv(os.path.join(save_path, f'NSE_total_{target_point}.csv'), index=False)

   

if __name__ == "__main__":

    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)