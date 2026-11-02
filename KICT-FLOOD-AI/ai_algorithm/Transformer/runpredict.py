from io import StringIO
import sys
import ast
import os
from datetime import timedelta


def auto_cast(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def normalize_dir(path: str) -> str:
    if path is None:
        return ""
    return os.path.abspath(os.path.expanduser(str(path).strip().strip('"').strip("'")))


def main():
    
    # 로그 파일에 파라메터 정보가 부족할 경우 오류를 바로 기록되도록 설정
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if len(sys.argv) < 3:
        print("[Model Creation] Invalid arguments.")
        print("Usage: python makemodel.py model_parameter.txt python_exe_path [predict_data_path]")
        sys.exit(1)

    datafilepath = sys.argv[1]
    modelParameterPath = sys.argv[2]
    predict_data_path = sys.argv[3]
    saveReportPath = normalize_dir(sys.argv[4])
    modelName = sys.argv[5]
    get_pypath = sys.argv[6]

    import subprocess

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
                sys.exit(1)

    ensure_package("tensorflow", "tensorflow")
    ensure_package("scikit-learn", "sklearn")
    ensure_package("numpy", "numpy")
    ensure_package("matplotlib", "matplotlib")
    ensure_package("pandas", "pandas")

    import numpy as np
    import tensorflow as tf
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pickle import load

    os.makedirs(saveReportPath, exist_ok=True)

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

    dataModelpath = normalize_dir(modeldict_data.get("save_path"))    
    save_path = saveReportPath
    dataset_filename = datafilepath
    model_name = modelName
    drop_out_rate = modeldict_data.get("drop_out_rate")
    seq_length = int(modeldict_data.get("seq_length"))
    training_rate = float(modeldict_data.get("training_rate"))
    test_start = float(modeldict_data.get("test_start"))
    watershed_with_target = modeldict_data.get("watershed_with_target")
    criteria_header = modeldict_data.get("criteria_header", []) or []
    criteria_wl = modeldict_data.get("criteria_wl", []) or []
    apply_realtime_Pred_WS = bool(modeldict_data.get("apply_realtime_Pred_WS", False))
    realtime_Pred_WS_time = int(modeldict_data.get("realtime_Pred_WS_time", 0) or 0) if apply_realtime_Pred_WS else 0

    Data_X_Column = modeldict_data.get("Data_X_Column")
    Data_Y_Column = modeldict_data.get("Data_Y_Column")
    if not isinstance(Data_X_Column, list) or not isinstance(Data_Y_Column, list):
        print("[Prediction and Performance Analysis] Data_X_Column/Data_Y_Column must be list.")
        sys.exit(1)

    data_dim = len(Data_X_Column)
    output_dim = len(Data_Y_Column)

    lead_time = modeldict_data.get("lead_time", "timeseries")
    Predict_time_leadtime = modeldict_data.get("Predict_time_leadtime")
    Predict_time_index = modeldict_data.get("Predict_time_index")
    if not isinstance(Predict_time_leadtime, list) or not isinstance(Predict_time_index, list):
        print("[Prediction and Performance Analysis] Predict_time_leadtime/Predict_time_index must be list.")
        sys.exit(1)

    target_point = dataset_filename[dataset_filename.find('Target_') + 7:dataset_filename.find('Target_') + 14]
    if 'Target_' not in dataset_filename:
        target_point = str(modeldict_data.get("model_name", "target"))

    if seq_length < 1:
        print("[Prediction and Performance Analysis] seq_length must be greater than or equal to 1.")
        sys.exit(1)

    # read input data
    data = pd.DataFrame()
    try:
        data = pd.read_csv(dataset_filename, parse_dates=['Date'])
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    except FileNotFoundError:
        print(f"[Prediction and Performance Analysis] Data file not found : {dataset_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to load data file : {e}")
        sys.exit(1)

    if data.empty:
        print("[Prediction and Performance Analysis] Dataset is empty.")
        sys.exit(1)

    # Y data를 수위 값이 아닌 수위변동 값으로 계산
    target_header = data.columns[1]
    Y_header = data.columns[Data_Y_Column]
    data.loc[:, Y_header] = data.loc[:, Y_header].sub(data[target_header], axis=0)

    train_size = int(len(data) * training_rate)
    test_size = len(data) - int(len(data) * test_start)

    data_training = data.iloc[:train_size].copy()
    data_test = data.iloc[int(len(data) * test_start):].copy()

    print('Total = ', len(data))
    print('Test = ', len(data_test))

    no_test_flag_path = os.path.join(saveReportPath, "no_test_data.flag")
    if os.path.exists(no_test_flag_path):
        try:
            os.remove(no_test_flag_path)
        except Exception:
            pass

    if data_test is None or len(data_test) == 0:
        print("[Prediction and Performance Analysis] There is no test data. Prediction could not be performed.")
        with open(no_test_flag_path, "w", encoding="utf-8") as f:
            f.write("NO_TEST_DATA\n")
        sys.exit(0)

    # data slicing
    try:
        X_data_training_drop = data_training.iloc[:, Data_X_Column]
        Y_data_training_drop = data_training.iloc[:, Data_Y_Column]
        X_data_test_drop = data_test.iloc[:, Data_X_Column]
        Y_data_test_drop = data_test.iloc[:, Data_Y_Column]
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to slice dataset columns : {e}")
        sys.exit(1)

    X_data_training_column_list = X_data_training_drop.columns.tolist()
    Y_data_training_column_list = Y_data_training_drop.columns.tolist()

    # load scalers created by makemodel.py
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

    # test process
    X_df = X_data_test_drop
    Y_df = Y_data_test_drop

    # 테스트 데이터가 없을 경우 모델 생성은 완료하고 평가만 건너뜀
    if len(data_test) == 0 or X_df.empty or Y_df.empty:
        print("[Model Creation] Model was created successfully. However, there was no test data, so prediction/evaluation was skipped.")
        try:
            with open(no_test_flag_path, "w", encoding="utf-8") as f:
                f.write("NO_TEST_DATA\n")
        except Exception as e:
            print(f"[Model Creation] Failed to create no_test_data.flag : {e}")
        sys.exit(0)

    # data normalize of test data
    X_inputs = X_scaler.transform(X_df)
    Y_inputs = Y_scaler.transform(Y_df)


    ## Find a Temporal discontinuity for test data

    DeltaT_test = []

    data_test['Date'] = pd.to_datetime(data_test['Date'], errors='coerce')
    data_test = data_test.dropna(subset=['Date']).reset_index(drop=True)

    for i in range(len(data_test)):
        if i == 0:
            temp_deltaT = 0
        else:
            temp_deltaT = int(
                (data_test['Date'].iloc[i] - data_test['Date'].iloc[i-1]).total_seconds() / 60
            )

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

    if X_test.shape[0] == 0 or Y_test.shape[0] == 0:
        print("[Model Creation] Model was created successfully. However, no valid continuous test sequences were found, so prediction/evaluation was skipped.")
        try:
            with open(no_test_flag_path, "w", encoding="utf-8") as f:
                f.write("NO_VALID_TEST_SEQUENCE\n")
        except Exception as e:
            print(f"[Model Creation] Failed to create no_test_data.flag : {e}")
        sys.exit(0)

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

    # load the best model
    try:
        new_model = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))
    except OSError:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} File not found : ")
        sys.exit(1)
    except Exception as e:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} Failed to load model : {e}")
        sys.exit(1)

    # calculate predict Y
    try:
        Y_pred = new_model.predict(X_test)
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

    # Y data를 수위변동 값에서 수위값으로 계산
    result_column = result_all.columns
    Y_header=[]

    for j in range(len(result_column)):
        if 'Y_' in result_column[j] :
            Y_header.append(result_column[j])

    for j in range(0,len(Y_header),2) :
        result_all[Y_header[j]] = eval('result_all.{}+result_all.{}'.format(Y_header[j],target_header))
        result_all[Y_header[j+1]] = eval('result_all.{}+result_all.{}'.format(Y_header[j+1],target_header))

    result_all.to_csv(os.path.join(save_path, f'predict_result_{model_name}_timeseries_drop-out_{drop_out_rate}_seq_length_{seq_length}.csv'), index=False)

    
    # Evaluating process
    import math


    # Analysis of each Criteria waterlevel

    NSE_total = pd.DataFrame()

    for n in range(len(criteria_wl)+1) :
        nse = list()

        # 해당 기준수위 조건에 실제 자료가 하나라도 있는지 확인
        has_criteria_data = False

        if n==0 :
            save_graph_path = save_path

        else :
            save_graph_path = os.path.join(save_path, f'over_{criteria_header[n-1]}')

        # 리드타임별 NSE 계산
        for i in range(len(Predict_time_leadtime)):
            tmp_data_filter = result_all[
                [
                    'Y_true_{}'.format(Predict_time_index[i]),
                    'Y_pred_{}'.format(Predict_time_index[i])
                ]
            ].copy()

            # 기준수위별 NSE는 각 리드타임 실제수위 기준으로 자료를 선택
            if n!=0 :
                tmp_data_filter = tmp_data_filter.drop(
                    tmp_data_filter[
                        tmp_data_filter[
                            'Y_true_{}'.format(Predict_time_index[i])
                        ] < criteria_wl[n-1]
                    ].index
                )

            # 결측값은 NSE 계산에서 제외
            tmp_data_filter = tmp_data_filter.dropna().reset_index(drop=True)

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
            if len(Y_true_filter) == 0 :
                obs_avg = None
                nse.append(np.nan)

            else :
                obs_avg = sum(Y_true_filter, 0.0) / len(Y_true_filter)

                e1 = list()
                e2 = list()

                for j in range(0,len(Y_true_filter)):
                    e1.append((Y_true_filter[j]-Y_pred_filter[j])**2)
                    e2.append((Y_true_filter[j]-obs_avg)**2)

                sum_e1 = sum(e1)
                sum_e2 = sum(e2)

                # 관측수위가 모두 같은 값이면 NSE 계산 불가
                if sum_e2 == 0 :
                    nse.append(np.nan)

                else :
                    nse.append(1- sum_e1/sum_e2)

            # Visualising the results
            if len(Y_true_filter) > 0 :

                # 해당 기준수위 조건에 자료가 하나라도 있음을 기록
                has_criteria_data = True

                # 해당 기준수위 자료가 있을 때만 결과 폴더 생성
                if not os.path.exists(save_graph_path):
                    os.makedirs(save_graph_path)

                plt.figure(figsize=(14,5))
                plt.plot(Y_true_filter, color = 'red', label = 'Observed Water Level')
                plt.plot(Y_pred_filter, color = 'blue', label = 'Predicted Water Level')

                if n==0 :
                    plt.title(
                        'Water Level Prediction_lead time_{}(NSE ={})'.format(
                            Predict_time_index[i],
                            str(nse[i])
                        )
                    )

                else :
                    plt.title(
                        'Water Level Prediction_lead time_{}(NSE over {}={})'.format(
                            Predict_time_index[i],
                            criteria_header[n-1],
                            str(nse[i])
                        )
                    )

                plt.xlabel('Time')
                plt.ylabel('Water Level')
                plt.legend()

                if n==0 :
                    plt.savefig(
                        os.path.join(
                            save_graph_path,
                            f'prediction_result_{model_name}_after_{Predict_time_index[i]}_drop-out_{drop_out_rate}_seq_length_{seq_length}.png'
                        )
                    )

                else :
                    plt.savefig(
                        os.path.join(
                            save_graph_path,
                            'prediction_result_'+model_name+
                            '_after_'+Predict_time_index[i]+
                            '_drop-out_'+str(drop_out_rate)+
                            "_seq_length_"+str(seq_length)+'.png'
                        )
                    )

                plt.close()

        if n==0 :
            print('== NSE ==')
            print('NSE=',nse)

            NSE_total = pd.concat(
                [NSE_total,pd.DataFrame({'All_data':nse})],
                axis=1
            )
            NSE_total.to_csv(
                os.path.join(save_path, 'NSE_total_{}.csv'.format(target_point)),
                index=False
            )

        else :
            
            # 해당 기준수위에 실제 자료가 하나도 없는 경우
            if not has_criteria_data:                
                print('**No data for calculation of NSE over {}'.format(criteria_header[n-1]))
                continue

            # 실제 자료가 있는 경우에만 NSE 결과 출력 및 CSV 컬럼 저장
            print('== NSE over {} =='.format(criteria_header[n-1]))
            print('NSE=', nse)

            NSE_total = pd.concat(
                [NSE_total, pd.DataFrame({criteria_header[n-1]: nse})],
                axis=1
            )
            NSE_total.to_csv(
                os.path.join(save_path, 'NSE_total_{}.csv'.format(target_point)),
                index=False
            )


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
