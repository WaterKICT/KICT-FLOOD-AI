import ast
import os
import sys
from datetime import timedelta
from pickle import load


def auto_cast(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def read_model_param(model_parameter_path: str) -> dict:
    modeldict_data = {}
    last_error = None

    for enc in ("cp949", "utf-8", "utf-8-sig"):
        try:
            with open(model_parameter_path, "r", encoding=enc) as file:
                for line in file:
                    if "," not in line:
                        continue
                    key, value = line.strip().split(",", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "watershed_with_target":
                        modeldict_data[key] = value
                    else:
                        modeldict_data[key] = auto_cast(value)
            return modeldict_data
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except FileNotFoundError:
            print(f"{model_parameter_path} File not found : ")
            sys.exit(1)
        except Exception as e:
            print(f"{model_parameter_path} Error occurred while processing the file : {e}")
            sys.exit(1)

    print(f"{model_parameter_path} Failed to decode parameter file : {last_error}")
    sys.exit(1)


def ensure_folder(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_dir(path: str) -> str:
    return os.path.normpath(path) if path else path


def make_no_test_flag(save_report_path: str, message: str):
    ensure_folder(save_report_path)
    no_test_flag_path = os.path.join(save_report_path, "no_test_data.flag")
    try:
        with open(no_test_flag_path, "w", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to create no_test_data.flag : {e}")


def build_valid_sequence_indices(data_test, seq_length: int, required_gap_minutes: int):
    import numpy as np

    if seq_length < 1:
        print("[Prediction and Performance Analysis] seq_length must be greater than or equal to 1.")
        sys.exit(1)

    if seq_length == 1:
        return np.arange(len(data_test), dtype=int)

    delta_minutes_test = (
        data_test["Date"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    rolling_gap_sum_test = delta_minutes_test.rolling(
        window=seq_length - 1,
        min_periods=seq_length - 1
    ).sum()

    return np.flatnonzero(
        np.isclose(rolling_gap_sum_test.to_numpy(), required_gap_minutes)
    )


def main():
    if len(sys.argv) < 7:
        print("[Prediction and Performance Analysis] Invalid arguments.")
        print("Usage: python runpredict.py datafilepath modelParameterPath predRainPath saveReportPath modelName python_exe_path")
        sys.exit(1)

    datafilepath = sys.argv[1]
    modelParameterPath = sys.argv[2]
    predRainPath = sys.argv[3]
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
    import pandas as pd
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_folder(saveReportPath)

    modeldict_data = read_model_param(modelParameterPath)

    dataModelpath = normalize_dir(str(modeldict_data.get("save_path", "")))
    if not dataModelpath:
        print("[Prediction and Performance Analysis] save_path is empty in modelParam.txt.")
        sys.exit(1)

    # modelParam.txt의 값이 우선이지만, 실행 인자로 받은 모델명/데이터셋/저장경로를 사용한다.
    dataset_filename = datafilepath
    model_name = modelName or str(modeldict_data.get("model_name", ""))
    save_path = saveReportPath

    drop_out_rate = modeldict_data.get("drop_out_rate")
    seq_length = int(modeldict_data.get("seq_length"))
    training_rate = float(modeldict_data.get("training_rate", 1.0))
    test_start = float(modeldict_data.get("test_start", 0.7))

    watershed_with_target = modeldict_data.get("watershed_with_target")
    criteria_header = modeldict_data.get("criteria_header", [])
    criteria_wl = modeldict_data.get("criteria_wl", [])
    apply_realtime_Pred_WS = bool(modeldict_data.get("apply_realtime_Pred_WS", False))
    realtime_Pred_WS_time = int(modeldict_data.get("realtime_Pred_WS_time", 0)) if apply_realtime_Pred_WS else 0

    Data_X_Column = modeldict_data.get("Data_X_Column")
    Data_Y_Column = modeldict_data.get("Data_Y_Column")
    Predict_time_leadtime = modeldict_data.get("Predict_time_leadtime")
    Predict_time_index = modeldict_data.get("Predict_time_index")

    if not isinstance(Data_X_Column, list) or not isinstance(Data_Y_Column, list):
        print("[Prediction and Performance Analysis] Data_X_Column/Data_Y_Column must be list in modelParam.txt.")
        sys.exit(1)

    if not isinstance(Predict_time_leadtime, list) or not isinstance(Predict_time_index, list):
        print("[Prediction and Performance Analysis] Predict_time_leadtime/Predict_time_index must be list in modelParam.txt.")
        sys.exit(1)

    data_dim = len(Data_X_Column)
    output_dim = len(Data_Y_Column)
    lead_time = str(modeldict_data.get("lead_time", "timeseries"))

    # Target 관측소 코드는 파일명에서 추출하되 실패하면 model_name 기준으로 최대한 유지
    target_pos = dataset_filename.find("Target_")
    target_point = dataset_filename[target_pos + 7:target_pos + 14] if target_pos >= 0 else str(modeldict_data.get("target_point", ""))

    # 데이터셋 읽기
    try:
        data = pd.read_csv(dataset_filename, parse_dates=["Date"])
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    except FileNotFoundError:
        print(f"[Prediction and Performance Analysis] Data file not found : {dataset_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to load data file : {e}")
        sys.exit(1)

    if data.empty:
        print("[Prediction and Performance Analysis] Dataset is empty.")
        make_no_test_flag(save_path, "EMPTY_DATASET")
        sys.exit(0)

    target_header = data.columns[1]
    Y_header = data.columns[Data_Y_Column]

    train_size = int(len(data) * training_rate)
    test_size = len(data) - int(len(data) * test_start)

    data_training = data.iloc[:train_size].copy()
    data_test = data.iloc[int(len(data) * test_start):].copy().reset_index(drop=True)

    print("Total = ", len(data))
    print("Test = ", len(data_test))

    if len(data_test) == 0:
        print("[Prediction and Performance Analysis] There is no test data. Prediction could not be performed.")
        make_no_test_flag(save_path, "NO_TEST_DATA")
        sys.exit(0)

    try:
        X_data_training_drop = data_training.iloc[:, Data_X_Column]
        X_data_test_drop = data_test.iloc[:, Data_X_Column]
        Y_data_test_drop = data_test.iloc[:, Data_Y_Column]
    except Exception as e:
        print(f"[Prediction and Performance Analysis] Failed to slice X/Y columns : {e}")
        sys.exit(1)

    # modelParam.txt에 저장된 헤더를 우선 사용하고, 없으면 데이터셋에서 재생성
    X_data_training_column_list = modeldict_data.get("X_data_training_column_list")
    if not isinstance(X_data_training_column_list, list) or not X_data_training_column_list:
        X_data_training_column_list = X_data_training_drop.columns.tolist()

    x_scaler_path = os.path.join(dataModelpath, "X_scaler.pkl")
    y_scaler_path = os.path.join(dataModelpath, "Y_scaler.pkl")

    try:
        with open(x_scaler_path, "rb") as f:
            X_scaler = load(f)
        with open(y_scaler_path, "rb") as f:
            Y_scaler = load(f)
    except FileNotFoundError as fe:
        print(f"{fe}")
        sys.exit(1)
    except Exception as e:
        print(f"{e}")
        sys.exit(1)


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
                exist_rain_file = os.path.join(predRainPath, '{}.csv'.format(post_list[n][12:]))
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
        new_model = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))
    except OSError:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} File not found : ")
        sys.exit(1)
    except Exception as e:
        print(f"{os.path.join(dataModelpath, 'best_model.h5')} Failed to load model : {e}")
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

        print('== NSE over {} =='.format(criteria_header[n]))
        print('NSE=',nse_filter)

        NSE_total = pd.concat([NSE_total,pd.DataFrame({criteria_header[n]:nse_filter})],axis=1)
        NSE_total.to_csv(os.path.join(save_path, f'NSE_total_{target_point}.csv'), index=False)


if __name__ == "__main__":
    main()
