
from io import StringIO
import sys

def main():

    import subprocess

    get_pypath = sys.argv[2]

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

    modelPath = sys.argv[1]

    # 파일 존재 여부 확인
    if not os.path.isfile(modelPath):
        print(f"{modelPath} not found.", file=sys.stderr)
        sys.exit(1)  # 필요시 프로그램 종료
    
    # data normalize of training data    
    from pickle import load
    
    folder = os.path.dirname(modelPath)  
    x_scaler_path = os.path.join(folder, 'X_scaler.pkl')
    y_scaler_path = os.path.join(folder, 'Y_scaler.pkl')

    try:
        with open(x_scaler_path, 'rb') as f:
            X_scaler = load(f)

        with open(y_scaler_path, 'rb') as f:
            Y_scaler = load(f)

    except FileNotFoundError as fe:
        print(f"{fe}", file=sys.stderr)        
        sys.exit(1)
    except Exception as e:
        print(f"{e}", file=sys.stderr)        
        sys.exit(1)

    # 모델 로딩 시 예외 처리
    try:
        modelInfo = tf.keras.models.load_model(modelPath)
    except OSError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

    # 모델 요약 출력
    modelInfo.summary()

    # 모델 종류 확인
    layer_names = [layer.__class__.__name__ for layer in modelInfo.layers]

    print("\n==============================")
    print("Model layer list:")
    for name in layer_names:
        print(" -", name)

    print("==============================")

    if "GRU" in layer_names:
        model_type = "GRU"

    elif "MultiHeadAttention" in layer_names:
        model_type = "Transformer"

    elif "Conv1D" in layer_names and "LSTM" in layer_names:
        #model_type = "Conv1D + LSTM"
        model_type = "Conv-1D"

    elif "Conv1D" in layer_names:
        model_type = "Conv-1D"

    elif "LSTM" in layer_names:
        model_type = "LSTM"

    else:
        model_type = "Unknown"

    print(f"Detected model type: {model_type}")
    print("==============================\n")
    
if __name__ == '__main__':
    main()
