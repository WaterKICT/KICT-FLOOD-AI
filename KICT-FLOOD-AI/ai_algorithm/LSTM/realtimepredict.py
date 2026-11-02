from io import StringIO
import sys
    
def main():

    import subprocess

    get_pypath = sys.argv[8]   

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

    import tensorflow as tf
    import os
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # load x, y scaler
    from pickle import load

    dataModelpath = sys.argv[1]    
    random_number = int(sys.argv[2])
    seq_length = int(sys.argv[3])
    data_dim = int(sys.argv[4])
    stdTime = datetime.strptime(sys.argv[5], "%Y-%m-%d %H:%M:%S")
    Predict_time =  datetime.strptime(sys.argv[6], "%Y-%m-%d %H:%M:%S")
    resultFileName = sys.argv[7]
    
    data_test_drop = None
    input_csv = sys.stdin.read()
    if input_csv.strip():  # 공백이 아닌 경우에만 처리
        data_test_drop = pd.read_csv(StringIO(input_csv))
    else:
        print("Received empty data!")
    
    # data normalize of test data
    scaler = load(open(dataModelpath + '/X_scaler.pkl', 'rb'))  
    Y_scaler = load(open(dataModelpath + '/Y_scaler.pkl', 'rb'))

    tf.random.set_seed(random_number)  # for reproducibility

    # load_model
    try:
        savedModel = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))
    except Exception as e:
        print(f"[Realtime Prediction] Failed to load model : {e}")
        sys.exit(1)

    # test process
    df = data_test_drop

    # 입력 데이터 확인
    if df is None or df.empty:
        print("[Realtime Prediction] Input data is empty.")
        sys.exit(1)

    # scaler 입력 컬럼 개수 확인
    if hasattr(scaler, "n_features_in_") and df.shape[1] != scaler.n_features_in_:
        print(
            f"[Realtime Prediction] Input feature count mismatch. "
            f"Expected {scaler.n_features_in_}, got {df.shape[1]}"
        )
        sys.exit(1)


    # load scaler
    try:
        inputs = scaler.transform(df)
    except Exception as e:
        print(f"[Realtime Prediction] Failed to scale input data : {e}")
        sys.exit(1)

    # 입력 길이 확인
    if inputs.shape[0] <= seq_length:
        print(
            f"[Realtime Prediction] Not enough input rows. "
            f"Need more than seq_length={seq_length}, got {inputs.shape[0]}"
        )
        sys.exit(1)

    X_test = []

    # make dataset of test
    for i in range(seq_length, inputs.shape[0]):
        X_test.append(inputs[i - seq_length:i, 0:data_dim])

    # data normalize of test data
    X_test = np.array(X_test)

    if X_test.size == 0:
        print("[Realtime Prediction] X_test is empty. Prediction cannot be performed.")
        sys.exit(1)

    try:
        Y_pred = savedModel.predict(X_test)
        Y_pred = Y_scaler.inverse_transform(Y_pred)

         # make the result file
        result = pd.DataFrame({"date_time": str(stdTime), "y_date" : Predict_time, "y_pred": Y_pred[:, 0]})
        if not os.path.isfile(resultFileName):
            result.to_csv(resultFileName, index=False)
        else:
            result.to_csv(resultFileName, mode='a', index=False, header=False)
    except Exception as e:
        print(f"[Realtime Prediction] Error occurred during prediction or inverse transformation : {e}")
        Y_pred = None   
        sys.exit(1)

if __name__ == '__main__':
    #main()
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
