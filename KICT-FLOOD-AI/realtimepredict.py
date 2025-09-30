from io import StringIO
import sys
import time
    
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
    from sklearn.preprocessing import MinMaxScaler
    import os
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec    
    import matplotlib.ticker as ticker  
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
    savedModel = tf.keras.models.load_model(os.path.join(dataModelpath, 'best_model.h5'))

    # Training process
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

    # test process
    df = data_test_drop

    # load scaler
    inputs = scaler.transform(df)

    X_test = []

    # make dataset of test
    for i in range(seq_length, inputs.shape[0]):
        X_test.append(inputs[i - seq_length:i, 0:data_dim])

    # data normalize of test data
    X_test = np.array(X_test)

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
        print(f"Error occurred during prediction or inverse transformation : {e}")
        Y_pred = None   

if __name__ == '__main__':
    main()
