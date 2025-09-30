
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
            print(f"{pkg_name} 가 설치되어 있지 않습니다. 설치를 진행합니다...")
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
    #X_scaler = load(open(dataModelpath + '/X_scaler.pkl', 'rb'))
    #Y_scaler = load(open(dataModelpath + '/Y_scaler.pkl', 'rb'))
    # 상위 폴더
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
    
if __name__ == '__main__':
    main()
