import sys
import ast
import subprocess
import os
import time
from random import randrange
from pickle import dump

def auto_cast(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    
def main():

    # 로그 파일에 print/에러가 바로 기록되도록 설정
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if len(sys.argv) < 3:
        print("[Bayesian Optimization] Invalid arguments.")
        print("Usage: python bayesian.py optimization_param.txt python_exe_path [predict_data_path]")
        sys.exit(1)

    model_parameter = sys.argv[1]
    get_pypath = sys.argv[2]
    predict_data_path = sys.argv[3] if len(sys.argv) >= 4 else ""

    modeldict_data = {}

    try:
        with open(model_parameter, "r", encoding="cp949") as file:
            for line in file:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    key = key.strip()
                    value = value.strip()
                    modeldict_data[key] = auto_cast(value)
    except FileNotFoundError:
        print(f"[Bayesian Optimization] Parameter file not found : {model_parameter}")
        sys.exit(1)
    except Exception as e:
        print(f"[Bayesian Optimization] Failed to load parameter file : {e}")
        sys.exit(1)

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
    ensure_package("optuna", "optuna")
    
    # make_model
    import optuna
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # =====================================================
    # [추가] Bayesian Optimization Setting
    # True: Bayesian Optimization으로 최적 하이퍼파라미터 탐색 후 CSV 저장
    # False: 저장된 CSV에서 최적 파라미터를 불러와 기존 방식으로 풀 학습
    # =====================================================

    best_params_file = 'best_params.csv'    

    bayesian_epoch = int(modeldict_data.get("bayesian_epoch", 30))
    testing_epoch = int(modeldict_data.get("testing_epoch", 15))

    # 최적화 방식 : range(범위 탐색) 또는 list(후보값 중 선택)
    search_mode = str(modeldict_data.get("search_mode", "range")).strip().lower()

    if search_mode not in ("range", "list"):
        search_mode = "range"

    # 범위 방식(Min / Max) 값 읽기
    seq_length_range = [
        int(modeldict_data.get("seq_length_min", 24)),
        int(modeldict_data.get("seq_length_max", 36))
    ]

    drop_out_rate_range = [
        float(modeldict_data.get("dropout_rate_min", 0.05)),
        float(modeldict_data.get("dropout_rate_max", 0.30))
    ]

    hidden_neuron_range = [
        int(modeldict_data.get("hidden_neuron_min", 16)),
        int(modeldict_data.get("hidden_neuron_max", 32))
    ]

    n_layers_range = [
        int(modeldict_data.get("number_of_layer_min", 2)),
        int(modeldict_data.get("number_of_layer_max", 4))
    ]

    batch_size_range = [
        int(modeldict_data.get("batch_size_min", 16)),
        int(modeldict_data.get("batch_size_max", 48))
    ]

    # Min/Max 역전 방지
    seq_length_range = sorted(seq_length_range)
    drop_out_rate_range = sorted(drop_out_rate_range)
    hidden_neuron_range = sorted(hidden_neuron_range)
    n_layers_range = sorted(n_layers_range)
    batch_size_range = sorted(batch_size_range)

    # 리스트 방식 입력값을 숫자 리스트로 변환한다.
    def get_search_list(key, default_values, value_type):
        values = modeldict_data.get(key, default_values)

        if not isinstance(values, (list, tuple)):
            values = default_values

        try:
            result = [value_type(value) for value in values]
        except (TypeError, ValueError):
            result = []

        return result if result else default_values

    # 리스트 방식(List) 탐색 후보값 읽기
    seq_length_list = get_search_list(
        "seq_length_list",
        [12, 18, 24, 36, 48, 72],
        int
    )

    drop_out_rate_list = get_search_list(
        "dropout_rate_list",
        [0.0, 0.05, 0.1, 0.2, 0.3],
        float
    )

    hidden_neuron_list = get_search_list(
        "hidden_neuron_list",
        [16, 32, 64],
        int
    )

    n_layers_list = get_search_list(
        "number_of_layer_list",
        [2, 3, 4],
        int
    )

    batch_size_list = get_search_list(
        "batch_size_list",
        [16, 32, 40, 48],
        int
    )

    # min/max 역전 방지
    seq_length_range = sorted(seq_length_range)
    drop_out_rate_range = sorted(drop_out_rate_range)
    hidden_neuron_range = sorted(hidden_neuron_range)
    n_layers_range = sorted(n_layers_range)
    batch_size_range = sorted(batch_size_range)

    # =====================================================
    # Training Parameter
    # =====================================================
    dataset_filename = modeldict_data.get("data_path", "")
    model_name = modeldict_data.get("model_name", "WL_timeseries")
    save_path = modeldict_data.get("save_path", "")

    if not dataset_filename or not os.path.isfile(dataset_filename):
        print(f"[Bayesian Optimization] Dataset file not found : {dataset_filename}")
        sys.exit(1)

    if not save_path:
        print("[Bayesian Optimization] Save path is empty.")
        sys.exit(1)

    save_path = os.path.join(save_path, "")

    training_rate = float(modeldict_data.get("training_rate", 1.0))
    #test_start = float(modeldict_data.get("test_start", 0.7))
    hidden_dim = int(modeldict_data.get("hidden_dim", 48))

    Data_X_Column = modeldict_data.get("Data_X_Column", [])
    Data_Y_Column = modeldict_data.get("Data_Y_Column", [])

    if not isinstance(Data_X_Column, list):
        print("[Bayesian Optimization] Data_X_Column must be list.")
        sys.exit(1)

    if not isinstance(Data_Y_Column, list):
        print("[Bayesian Optimization] Data_Y_Column must be list.")
        sys.exit(1)

    if not Data_X_Column:
        print("[Bayesian Optimization] Data_X_Column is empty.")
        sys.exit(1)

    if not Data_Y_Column:
        print("[Bayesian Optimization] Data_Y_Column is empty.")
        sys.exit(1)

    data_dim = len(Data_X_Column)
    output_dim = len(Data_Y_Column)

    validation_rate = float(modeldict_data.get("validation_rate", 0.2))
    validation_rate = min(max(validation_rate, 0.01), 0.5)

    activation_func = modeldict_data.get("activation_func", "relu")
    optimize_func = modeldict_data.get("optimize_func", "adam")
    loss_func = modeldict_data.get("loss_func", "mean_squared_error")

    # =====================================================
    # Data Loading & Preprocessing (기존 코드와 동일)
    # =====================================================

    # save the start time
    start = time.time()

    # make a random number
    random_number = randrange(1, 1000)
    tf.random.set_seed(random_number)

    # read input data
    try:
        data = pd.read_csv(dataset_filename, parse_dates=['Date'])
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    except FileNotFoundError:
        print(f"[Bayesian Optimization] Data file not found : {dataset_filename}")
        sys.exit(1)
    except Exception as e:
        print(f"[Bayesian Optimization] Failed to load data file : {e}")
        sys.exit(1)

    data = data.dropna(subset=['Date']).reset_index(drop=True)

    if data.empty:
        print("[Bayesian Optimization] Dataset is empty after Date parsing.")
        sys.exit(1)

    # rate of training data size
    train_size = int(len(data) * training_rate)
    #test_size = len(data) - int(len(data) * test_start)

    data_training = data[:train_size]
    #data_test = data[int(len(data) * test_start):]

    if data_training is None or len(data_training) == 0:
        print("[Bayesian Optimization] Training data is empty.")
        sys.exit(1)

    print('Total = ', len(data))
    print('Train = ', len(data_training))

    # data slicing
    max_col_index = len(data.columns) - 1
    invalid_x_cols = [idx for idx in Data_X_Column if idx < 0 or idx > max_col_index]
    invalid_y_cols = [idx for idx in Data_Y_Column if idx < 0 or idx > max_col_index]

    if invalid_x_cols:
        print(f"[Bayesian Optimization] Invalid Data_X_Column index : {invalid_x_cols}")
        sys.exit(1)

    if invalid_y_cols:
        print(f"[Bayesian Optimization] Invalid Data_Y_Column index : {invalid_y_cols}")
        sys.exit(1)

    X_data_training_drop = data_training.iloc[:, Data_X_Column]
    Y_data_training_drop = data_training.iloc[:, Data_Y_Column]

    X_data_training_column = pd.DataFrame({"header": X_data_training_drop.columns})
    X_data_training_column_list = list(X_data_training_column["header"])

    Y_data_training_column = pd.DataFrame({"header": Y_data_training_drop.columns})
    Y_data_training_column_list = list(Y_data_training_column["header"])

    # data normalize
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_data_training_scale = X_scaler.fit_transform(X_data_training_drop)
    Y_data_training_scale = Y_scaler.fit_transform(Y_data_training_drop)

    os.makedirs(save_path, exist_ok=True)

    # =====================================================
    # [추가] Function: 시간 불연속 체크 및 학습 데이터 생성
    # - 기존 코드의 시간 불연속 체크 + X_train/Y_train 생성 부분을 함수화
    # - Bayesian Optimization에서 seq_length가 매 trial마다 달라지므로
    #   seq_length를 인자로 받아 학습 데이터를 동적으로 생성하기 위해 함수로 분리
    # =====================================================
    def make_train_data(seq_len):
        """seq_length에 따라 X_train, Y_train을 생성하는 함수"""
        DeltaT = []
        for i in range(len(data_training)):
            if i == 0:
                temp_deltaT = 0
            else:
                temp_deltaT = int(((data_training['Date'].iloc[i] - data_training['Date'].iloc[i - 1]).seconds) / 60)
            DeltaT.append(temp_deltaT)

        df_DeltaT = pd.DataFrame({'DeltaT': DeltaT})

        check = []
        for n in range(len(data_training)):
            if n < (seq_len - 1):
                temp_check = ""
            else:
                temp_check = sum(df_DeltaT['DeltaT'][n + 2 - seq_len:n + 1])
                if temp_check == 10 * (seq_len - 1):
                    temp_check = n
                else:
                    temp_check = ""
            check.append(temp_check)

        X_train = []
        Y_train = []

        for k in check:
            if k == "":
                pass
            else:
                X_train.append(X_data_training_scale[k + 1 - seq_len:k + 1, :])
                Y_train.append(Y_data_training_scale[k, :])

        X_train, Y_train = np.array(X_train), np.array(Y_train)
        return X_train, Y_train

    # =====================================================
    # [추가] Function: 모델 생성
    # - 기존 코드의 모델 생성 부분(Sequential + LSTM 쌓기)을 함수화
    # - Bayesian Optimization에서 레이어 수/뉴런 수가 매번 달라지므로 함수로 분리
    # - 이 함수만 수정하면 GRU, Transformer 등 다른 아키텍처로 교체 가능
    # =====================================================
    def build_model(seq_len, n_layers, neurons_list, d_rate):

        model = Sequential()

        model.add(LSTM(units = hidden_dim, activation = activation_func, return_sequences = True, input_shape = (seq_len, data_dim)))
        model.add(Dropout(d_rate)) # prevent overfitting
        
        # add hidden layers 
        for i in range(0, n_layers):
            model.add(LSTM(units = neurons_list[i], activation = activation_func, return_sequences = True))
            model.add(Dropout(d_rate))  # prevent overfitting
            
        model.add(LSTM(units = hidden_dim, activation = activation_func))
        model.add(Dropout(d_rate))  # prevent overfitting

        model.add(Dense(units = output_dim))
        
        model.compile(optimizer=optimize_func, loss=loss_func, metrics=['acc'])
        return model

    # =====================================================
    # [추가] Bayesian Optimization (Optimization = True일 때 실행)
    # - 각 trial마다 testing_epoch만큼만 학습하여 val_loss를 평가
    # - 탐색 완료 후 최적 파라미터를 best_params.csv에 저장
    # =====================================================
    #if Optimization:
    print("\n" + "=" * 60)
    print(f"  Bayesian Epoch (n_trials): {bayesian_epoch}")
    print(f"  Testing Epoch: {testing_epoch}")

    # 선택한 탐색 방식에 따라 실제 적용되는 탐색 후보값을 출력한다.
    if search_mode == "list":
        print("  Search Mode: List")
        print(f"  Sequence Length List: {seq_length_list}")
        print(f"  Dropout Rate List: {drop_out_rate_list}")
        print(f"  Hidden Neuron List: {hidden_neuron_list}")
        print(f"  Number of Layers List: {n_layers_list}")
        print(f"  Batch Size List: {batch_size_list}")

    else:
        print("  Search Mode: Range")
        print(f"  Sequence Length Range: {seq_length_range}")
        print(f"  Dropout Rate Range: {drop_out_rate_range}")
        print(f"  Hidden Neuron Range: {hidden_neuron_range}")
        print(f"  Number of Layers Range: {n_layers_range}")
        print(f"  Batch Size Range: {batch_size_range}")
    print("=" * 60 + "\n")


    def objective(trial):
        # 탐색할 하이퍼파라미터 정의
        # 선택한 최적화 방식에 따라 범위 또는 후보 리스트에서 탐색한다.
        if search_mode == "list":
            trial_seq_length = trial.suggest_categorical(
                'seq_length',
                seq_length_list
            )

            trial_drop_out = trial.suggest_categorical(
                'drop_out_rate',
                drop_out_rate_list
            )

            trial_n_layers = trial.suggest_categorical(
                'number_of_layers',
                n_layers_list
            )

            trial_batch_size = trial.suggest_categorical(
                'batch_size',
                batch_size_list
            )

        else:
            trial_seq_length = trial.suggest_int(
                'seq_length',
                seq_length_range[0],
                seq_length_range[1]
            )

            trial_drop_out = trial.suggest_float(
                'drop_out_rate',
                drop_out_rate_range[0],
                drop_out_rate_range[1]
            )

            trial_n_layers = trial.suggest_int(
                'number_of_layers',
                n_layers_range[0],
                n_layers_range[1]
            )

            trial_batch_size = trial.suggest_int(
                'batch_size',
                batch_size_range[0],
                batch_size_range[1],
                step=8
            )        

        # 레이어 개수만큼 각 레이어의 Hidden Neuron 값을 탐색한다.
        trial_neurons_per_layer = []

        for layer_i in range(trial_n_layers):
            if search_mode == "list":
                n = trial.suggest_categorical(
                    f'hidden_neuron_L{layer_i + 1}',
                    hidden_neuron_list
                )
            else:
                n = trial.suggest_int(
                    f'hidden_neuron_L{layer_i + 1}',
                    hidden_neuron_range[0],
                    hidden_neuron_range[1],
                    step=8
                )

            trial_neurons_per_layer.append(n)

        # 학습 데이터 생성 (seq_length에 따라 달라짐)
        X_train, Y_train = make_train_data(trial_seq_length)

        if len(X_train) == 0:   #학습 데이터가 0개일 때만 trial을 실패 처리
            return float('inf')

        # 모델 생성 (n_layers=레이어 개수, neurons_list=각 레이어별 뉴런 수)
        model = build_model(trial_seq_length, trial_n_layers, trial_neurons_per_layer, trial_drop_out)

        # 학습 (testing_epoch만큼만)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)     
        try:
            history = model.fit(
                X_train,
                Y_train,
                epochs=int(testing_epoch),
                batch_size=int(trial_batch_size),
                validation_split=validation_rate,
                callbacks=[es],
                verbose=1
            )
        except Exception:
            import traceback
            print(f"[Bayesian Optimization] Trial {trial.number} model.fit failed.")
            traceback.print_exc()
            tf.keras.backend.clear_session()
            return float('inf')               

        # val_loss 반환
        best_val_loss = min(history.history['val_loss'])

        # GPU 메모리 정리
        tf.keras.backend.clear_session()

        print(f"  Trial {trial.number}: val_loss={best_val_loss:.6f} | "
            f"seq={trial_seq_length}, dropout={trial_drop_out:.3f}, "
            f"layers={trial_n_layers}, neurons={trial_neurons_per_layer}, "
            f"batch={trial_batch_size}")

        return best_val_loss

    # Optuna Study 실행
    study = optuna.create_study(direction='minimize')
    try:
        study.optimize(objective, n_trials=int(bayesian_epoch))
    except Exception:
        import traceback
        print("[Bayesian Optimization] study.optimize failed.")
        traceback.print_exc()
        sys.exit(1)

    # 최적 결과 출력
    print("\n" + "=" * 60)
    print("  Bayesian Optimization Complete")
    print("=" * 60)
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60 + "\n")

    # 최적 파라미터 CSV 저장
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df['best_val_loss'] = study.best_value

    best_params_path = os.path.join(save_path, best_params_file)
    best_params_df.to_csv(best_params_path, index=False)
    print(f"Best parameters saved to: {best_params_path}")

    # 전체 trial 결과 저장
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(save_path, 'optimization_trials.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f"All trial results saved to: {trials_path}")

    # Elapsed time
    Elapsed_time = int(time.time() - start)
    print(f"\nOptimization Elapsed time: {Elapsed_time}sec")   


if __name__ == "__main__":

    parameter_file_path = sys.argv[1] if len(sys.argv) >= 2 else ""
       
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 최적화 성공/실패 및 sys.exit() 여부와 관계없이 optimization_param.txt 삭제
        if parameter_file_path and os.path.isfile(parameter_file_path):
            try:
                os.remove(parameter_file_path)
                print(
                    f"[Bayesian Optimization] "
                    f"Temporary parameter file deleted: {parameter_file_path}"
                )
            except Exception as e:
                print(
                    f"[Bayesian Optimization] "
                    f"Failed to delete temporary parameter file: {e}"
                )