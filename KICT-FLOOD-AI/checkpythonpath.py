# -*- coding: utf-8 -*-

import json
import os
import platform
import re
import subprocess
import sys

# Python 검사 프로세스의 결과를 찾기 위한 구분 문자열
PROBE_MARKER = "__KICT_PYTHON_PROBE__="

# 각 Python 환경 검사 시 최대 대기 시간(초)
TIMEOUT_SECONDS = 90

# venv 검색 깊이. 사용자 폴더 전체를 무제한 검색하면 검사 시작이 매우 느려질 수 있어 제한한다.
VENV_SEARCH_MAX_DEPTH = 5

# 각 Python 실행 파일에서 독립적으로 실행할 환경 검사 코드
# 현재 프로그램을 실행한 Python이 아닌, 검사 대상 Python 내부의 패키지와 기능을 직접 확인한다.
PROBE_CODE = r'''
import json
import os
import platform
import subprocess
import sys
import tempfile

MARKER = "__KICT_PYTHON_PROBE__="
result = {
    "executable": sys.executable,
    "python_version": ".".join(str(x) for x in sys.version_info[:3]),
    "python_full": sys.version.replace("\n", " "),
    "architecture": platform.architecture()[0],
    "packages": {},
    "imports": {},
    "features": {},
}


# 설치된 패키지의 버전 확인
def distribution_version(names):
    try:
        try:
            from importlib import metadata
        except ImportError:
            import importlib_metadata as metadata
        for name in names:
            try:
                return metadata.version(name)
            except Exception:
                pass
    except Exception:
        pass
    return None


# KICT-FLOOD-AI에서 사용하는 주요 패키지 목록
package_names = {
    "TensorFlow": ["tensorflow", "tensorflow-intel", "tensorflow-cpu"],
    "NumPy": ["numpy"],
    "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn"],
    "matplotlib": ["matplotlib"],
    "Optuna": ["optuna"],
}
# 각 패키지의 설치 버전 확인
for display_name, dist_names in package_names.items():
    result["packages"][display_name] = distribution_version(dist_names)


# 실제 Import 가능 여부를 확인할 패키지 목록
imports = {
    "NumPy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "Optuna": "optuna",
    "TensorFlow": "tensorflow",
}
# 패키지를 실제로 Import하여 정상 사용 가능한지 확인
loaded = {}
for display_name, module_name in imports.items():
    try:
        module = __import__(module_name)
        loaded[module_name] = module
        result["imports"][display_name] = {"ok": True, "error": ""}
    except Exception as exc:
        result["imports"][display_name] = {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, str(exc)),
        }


# TensorFlow가 정상적으로 Import된 경우 best_model.h5 저장/재로드 검사
tf = loaded.get("tensorflow")
if tf is not None:
    # TensorFlow에서 실제 사용하는 tf.keras 버전 확인
    try:
        result["tf_keras_version"] = str(getattr(tf.keras, "__version__", "")) or None
    except Exception:
        result["tf_keras_version"] = None

    # Import 성공만으로는 DLL/연산 오류를 확인할 수 없으므로 실제 TensorFlow 연산을 수행한다.
    try:
        a = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        b = tf.constant([[3.0], [4.0]], dtype=tf.float32)
        value = float(tf.matmul(a, b).numpy()[0][0])
        if abs(value - 11.0) > 1e-6:
            raise RuntimeError("TensorFlow 계산 결과가 예상값과 다릅니다.")
        result["features"]["TensorFlow 실제 연산"] = {"ok": True, "error": ""}
    except Exception as exc:
        result["features"]["TensorFlow 실제 연산"] = {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, str(exc)),
        }

    # 실제 모형제작/예측수행에서 사용하는 best_model.h5 형식만 검사한다.
    h5_path = None
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3, 2)),
            tf.keras.layers.LSTM(2),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        fd, h5_path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        os.unlink(h5_path)
        model.save(h5_path)
        tf.keras.models.load_model(h5_path)
        result["features"]["best_model.h5 저장/재로드"] = {"ok": True, "error": ""}
    except Exception as exc:
        result["features"]["best_model.h5 저장/재로드"] = {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, str(exc)),
        }
    finally:
        if h5_path and os.path.isfile(h5_path):
            try:
                os.remove(h5_path)
            except Exception:
                pass

# 입력자료와 예측자료의 정규화 기준은 각각
# X_scaler.pkl, Y_scaler.pkl 파일로 저장한 후 예측수행 시 다시 불러온다.
# 현재 Python 환경에서 정규화 파일을 정상적으로 저장하고 불러올 수 있는지 확인한다.
try:
    import pickle
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    scaler_path = None
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.0], [1.0]], dtype=float))
    fd, scaler_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)
    loaded_scaler.transform(np.array([[0.5]], dtype=float))
    result["features"]["scaler 저장/재로드"] = {"ok": True, "error": ""}
except Exception as exc:
    result["features"]["scaler 저장/재로드"] = {
        "ok": False,
        "error": "%s: %s" % (type(exc).__name__, str(exc)),
    }
finally:
    if 'scaler_path' in locals() and scaler_path and os.path.isfile(scaler_path):
        try:
            os.remove(scaler_path)
        except Exception:
            pass

# makemodel.py 및 모델별 제작 코드에는 확장자 없는 model.save(...)가 있다.
# Keras 3 계열처럼 이 저장 방식 자체가 실패하는 환경인지 실제로 확인한다.
if tf is not None:
    save_dir = None
    try:
        save_dir = tempfile.mkdtemp()
        save_path = os.path.join(save_dir, "model_after_timeseries")
        test_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1),
        ])
        test_model.save(save_path)
        result["features"]["현재 model.save 방식"] = {"ok": True, "error": ""}
    except Exception as exc:
        result["features"]["현재 model.save 방식"] = {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, str(exc)),
        }
    finally:
        if save_dir:
            try:
                import shutil
                shutil.rmtree(save_dir)
            except Exception:
                pass

print(MARKER + json.dumps(result, ensure_ascii=False))
'''


# 외부 명령 실행
# Python 검색 및 환경 검사는 별도 프로세스로 실행하며 지정 시간 초과 시 실패 처리한다.
def run_command(command, timeout=15):
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            creationflags=creationflags,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except Exception:
        return -1, "", ""


# 검색된 Python 실행 파일을 검사 대상 목록에 추가
# 동일한 경로는 한 번만 추가하고 실제 파일이 존재하는 경우에만 등록한다.
def add_candidate(candidates, value):
    if not value:
        return
    value = value.strip().strip('"')
    if not value:
        return
    # py launcher 출력의 앞쪽 표시(-V:3.12 *, -3.8-64 등)를 제거한다.
    match = re.search(r"([A-Za-z]:[\\/].*?python(?:\.exe)?)\s*$", value, re.I)
    if match:
        value = match.group(1).strip()
    if os.name == "nt" and not value.lower().endswith(".exe"):
        return
    key = os.path.normcase(os.path.abspath(value))
    if os.path.isfile(value):
        candidates[key] = os.path.abspath(value)


# pyvenv.cfg가 있는 폴더를 venv로 보고 해당 환경의 Python 실행 파일을 추가
def add_venv_candidate(candidates, venv_dir):
    if not venv_dir or not os.path.isdir(venv_dir):
        return
    cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    if not os.path.isfile(cfg_path):
        return
    if os.name == "nt":
        executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        executable = os.path.join(venv_dir, "bin", "python")
    add_candidate(candidates, executable)


# 같은 폴더를 여러 검색 루트에서 반복 순회하지 않도록 상위/하위 중복 경로 정리
def compact_search_roots(paths):
    unique = []
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
        if not os.path.isdir(path):
            continue
        key = os.path.normcase(path)
        duplicate = False
        for saved_path, saved_key in unique:
            try:
                if os.path.commonpath([key, saved_key]) == saved_key:
                    duplicate = True
                    break
            except (ValueError, OSError):
                pass
        if not duplicate:
            unique.append((path, key))
    return [path for path, _ in unique]


# 사용자가 실제로 venv를 만들 가능성이 높은 위치를 제한 깊이로 자동 검색
def discover_venvs(candidates):
    search_roots = []

    # 현재 실행 중인 venv와 virtualenvwrapper 환경 경로
    for env_name in ("VIRTUAL_ENV", "WORKON_HOME"):
        value = os.environ.get(env_name)
        if value:
            search_roots.append(value)

    # 현재 작업 위치와 검사 파일 주변의 프로젝트 venv
    search_roots.extend([os.getcwd(), os.path.dirname(os.path.abspath(__file__))])

    if os.name == "nt":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            # 일반적으로 많이 사용하는 venv 저장 위치
            search_roots.extend([
                os.path.join(user_profile, ".virtualenvs"),
                os.path.join(user_profile, "venvs"),
                os.path.join(user_profile, "Envs"),
                os.path.join(user_profile, "Documents"),
            ])

        # 이 프로젝트의 호환성 테스트용으로 권장한 기본 위치
        system_drive = os.environ.get("SystemDrive", "C:")
        search_roots.append(os.path.join(system_drive + os.sep, "KICT_PYTHON_TEST"))

    # 필요하면 세미콜론(;)으로 검색 루트를 추가할 수 있다.
    extra_roots = os.environ.get("KICT_VENV_ROOTS", "")
    if extra_roots:
        search_roots.extend(path for path in extra_roots.split(os.pathsep) if path.strip())

    skip_names = {
        ".git", "node_modules", "AppData", "$Recycle.Bin", "System Volume Information",
        "Windows", "Program Files", "Program Files (x86)", "__pycache__",
    }

    for root in compact_search_roots(search_roots):

        # 검색 루트 자체가 venv일 수도 있다.
        add_venv_candidate(candidates, root)

        base_depth = root.rstrip(os.sep).count(os.sep)
        try:
            for current, dirs, files in os.walk(root):
                depth = current.rstrip(os.sep).count(os.sep) - base_depth
                dirs[:] = [d for d in dirs if d not in skip_names]
                if depth >= VENV_SEARCH_MAX_DEPTH:
                    dirs[:] = []
                if "pyvenv.cfg" in files:
                    add_venv_candidate(candidates, current)
                    # venv 내부의 site-packages까지 내려갈 필요가 없다.
                    dirs[:] = []
        except (OSError, PermissionError):
            pass


# PC에 설치된 Python 실행 파일 검색
# 일반 Python은 Launcher/PATH/Registry에서, venv는 pyvenv.cfg를 기준으로 별도 검색한다.
def discover_pythons():
    candidates = {}

    # 현재 이 검사 파일을 실행한 Python 추가
    add_candidate(candidates, sys.executable)

    if os.name == "nt":
        # Python Launcher가 알고 있는 모든 설치본.
        for args in (["py", "-0p"], ["py", "--list-paths"]):
            code, stdout, _ = run_command(args)
            if code == 0:
                for line in stdout.splitlines():
                    add_candidate(candidates, line)

        # PATH에 잡혀 있는 python.exe.
        code, stdout, _ = run_command(["where", "python"])
        if code == 0:
            for line in stdout.splitlines():
                add_candidate(candidates, line)

        # PythonCore에 등록된 실행 파일. 사용자/시스템 설치를 모두 본다.
        registry_roots = [
            r"HKCU\Software\Python\PythonCore",
            r"HKLM\Software\Python\PythonCore",
            r"HKLM\Software\WOW6432Node\Python\PythonCore",
        ]
        for root in registry_roots:
            code, stdout, _ = run_command(["reg", "query", root, "/s", "/v", "ExecutablePath"])
            if code == 0:
                for line in stdout.splitlines():
                    match = re.search(r"ExecutablePath\s+REG_\w+\s+(.+)$", line, re.I)
                    if match:
                        add_candidate(candidates, match.group(1))

    # Python Launcher에는 보통 venv가 표시되지 않으므로 추가로 검색한다.
    discover_venvs(candidates)

    return list(candidates.values())


# 지정한 Python 환경 검사
# 해당 Python으로 PROBE_CODE를 실행하여 패키지, Import, 실제 모형 기능을 확인한다.
def probe_python(executable):
    code, stdout, stderr = run_command([executable, "-c", PROBE_CODE], TIMEOUT_SECONDS)
    # 검사 결과 문자열(PROBE_MARKER)을 찾아 JSON 데이터로 변환
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            try:
                payload = json.loads(line[len(PROBE_MARKER):])
            except Exception:
                payload = None
            break
    # Python 실행 실패 또는 정상적인 검사 결과를 받지 못한 경우
    if payload is None:
        return {
            "executable": executable,
            "probe_error": True,
            "returncode": code,
            "error": (stderr or stdout or "검사 프로세스에서 결과를 받지 못했습니다.").strip()[-1000:],
        }
    payload["probe_error"] = False
    return payload


# 버전 문자열에서 주 버전(major version)만 추출
# 예: 3.12.4 -> 3
def major(version):
    if not version:
        return None
    match = re.match(r"(\d+)", str(version))
    return int(match.group(1)) if match else None


# 검사 결과를 PASS / FAIL 문자열로 변환
def yesno(value):
    return "PASS" if value else "FAIL"


# 검사 결과를 기준으로 현재 KICT-FLOOD-AI 코드와의 호환성 분석
def analyze(result):
    packages = result.get("packages", {})
    imports = result.get("imports", {})
    features = result.get("features", {})
    warnings = []
    notices = []
    blockers = []

    # 모형제작 및 예측수행에 필요한 필수 패키지 확인
    required = ["TensorFlow", "NumPy", "pandas", "scikit-learn", "matplotlib"]
    missing = []
    broken = []
    for name in required:
        state = imports.get(name, {})
        if not state.get("ok"):
            if packages.get(name) is None:
                missing.append(name)
            else:
                broken.append("%s (%s)" % (name, state.get("error", "Import 실패")))

    # 미설치 패키지는 프로그램 실행 시 자동으로 설치되므로 오류에서 제외한다.
    # 패키지가 설치되어 있는데도 불러오지 못한 경우만 실제 오류로 처리한다.
    if missing:
        notices.append(
            "프로그램 실행 시 자동 설치 대상: " + ", ".join(missing)
        )
    if broken:
        blockers.append("설치되어 있으나 Import 실패: " + "; ".join(broken))

    # TensorFlow를 불러온 후 실제 계산까지 정상적으로 수행되는지 확인한다.
    tf_state = features.get("TensorFlow 실제 연산")
    if tf_state is not None and not tf_state.get("ok"):
        blockers.append("TensorFlow Import는 성공했지만 실제 연산에 실패했습니다.")

    h5_state = features.get("best_model.h5 저장/재로드")
    if h5_state is not None and not h5_state.get("ok"):
        error = h5_state.get("error", "")
        if "Could not locate function 'mse'" in error:
            blockers.append(
                "학습 모델 파일(best_model.h5)을 불러올 수 없습니다. "
                "현재 Keras 버전과 H5 파일 형식의 호환성을 확인하세요."
            )
        else:
            blockers.append(
                "학습 모델 파일(best_model.h5)을 저장하거나 다시 불러오지 못했습니다."
            )

    # NumPy와 scikit-learn이 정상적으로 설치된 경우에만 scaler 검사 결과를 판정한다.
    # 관련 패키지가 미설치된 경우에는 프로그램 실행 시 설치되므로 오류로 표시하지 않는다.
    scaler_state = features.get("scaler 저장/재로드")
    scaler_dependencies_ready = (
        imports.get("NumPy", {}).get("ok")
        and imports.get("scikit-learn", {}).get("ok")
    )
    if (
        scaler_dependencies_ready
        and scaler_state is not None
        and not scaler_state.get("ok")
    ):
        blockers.append(
            "데이터 정규화 파일(X_scaler.pkl / Y_scaler.pkl)을 저장하거나 "
            "다시 불러오지 못했습니다. 모형제작 또는 예측수행이 정상적으로 "
            "동작하지 않을 수 있습니다."
        )

    save_state = features.get("현재 model.save 방식")
    if save_state is not None and not save_state.get("ok"):
        blockers.append(
            "학습 완료 후 모델을 저장하는 방식이 현재 Keras 버전과 호환되지 않습니다."
        )

    # Bayesian Optimization에 사용하는 Optuna 확인
    if packages.get("Optuna") is None:
        notices.append(
            "Optuna는 프로그램 실행 시 자동으로 설치됩니다. "
            "설치가 완료되면 Bayesian Optimization을 사용할 수 있습니다."
        )
    elif not imports.get("Optuna", {}).get("ok"):
        warnings.append(
            "Optuna가 설치되어 있지만 정상적으로 불러오지 못했습니다. "
            "일반 모형제작은 가능하지만 Bayesian Optimization은 사용할 수 없습니다."
        )

    # 미설치 상태, 실제 Import/기능 오류, 정상 상태를 구분한다.
    core_ready = all(imports.get(name, {}).get("ok") for name in required)
    if blockers:
        status = "현재 코드 호환 오류"
        summary = "학습 또는 저장/재사용 단계에서 실패 가능성이 있어 현재 코드 그대로는 권장하지 않습니다."
    elif not core_ready:
        status = "설치 후 확인 필요"
        summary = "프로그램 실행 시 패키지 설치 후 기능 호환성이 확인됩니다."
    else:
        status = "사용 가능"
        summary = "현재 환경에서 best_model.h5 저장/재로드 검사를 통과했습니다."

    return status, summary, blockers, warnings, notices


# 하나의 Python 환경에 대한 상세 검사 결과 출력
def print_environment(index, total, result):
    print("\n" + "=" * 78)
    print("[%d/%d] Python 환경 검사" % (index, total))
    print("=" * 78)
    print("경로   : %s" % result.get("executable", "알 수 없음"))

    # Python 자체를 실행하지 못했거나 검사 결과를 받지 못한 경우
    if result.get("probe_error"):
        print("결과   : [검사 불가]")
        print("원인   : Python을 정상적으로 실행할 수 없습니다.")
        print("상세   : %s" % result.get("error", "알 수 없음"))
        return "검사 불가"

    # Python 기본 정보 출력
    print("버전   : Python %s (%s)" % (result.get("python_version", "?"), result.get("architecture", "?")))
    print("\n[주요 패키지]")
    print("%-18s %-16s %-10s" % ("항목", "버전", "상태"))
    print("-" * 78)
    for name in ["TensorFlow", "NumPy", "pandas", "scikit-learn", "matplotlib", "Optuna"]:
        version = result.get("packages", {}).get(name) or "미설치"
        import_state = result.get("imports", {}).get(name)
        if import_state is None:
            state, error = "-", ""
        else:
            state = yesno(import_state.get("ok"))
            error = import_state.get("error", "") if not import_state.get("ok") else ""
        # 정상 여부를 한눈에 확인할 수 있도록 상세 오류는 표에서 제외한다.
        if version == "미설치":
            display_state = "자동 설치"
        else:
            display_state = {"PASS": "정상", "FAIL": "오류", "INFO": "확인", "-": "-"}.get(state, state)
        print("%-18s %-16s %-10s" % (name, version, display_state))

    # 실제 프로그램에서 사용하는 계산, 파일 저장 및 불러오기 기능 확인
    if result.get("features"):
        print("\n[실행 기능]")
        feature_names = [
            ("TensorFlow 실제 연산", "TensorFlow 실제 계산"),
            ("best_model.h5 저장/재로드", "학습 모델 파일 저장/불러오기"),
            ("scaler 저장/재로드", "데이터 정규화 파일 저장/불러오기"),
            ("현재 model.save 방식", "학습 완료 후 모델 저장"),
        ]
        for name, display_name in feature_names:
            state = result.get("features", {}).get(name)
            if state is None:
                continue
            imports = result.get("imports", {})
            deferred = (
                name == "scaler 저장/재로드"
                and not (
                    imports.get("NumPy", {}).get("ok")
                    and imports.get("scikit-learn", {}).get("ok")
                )
            )
            if deferred:
                display_state = "설치 후 검사"
            else:
                display_state = "정상" if state.get("ok") else "오류"
            print("%-34s %s" % (display_name, display_state))

    # 현재 KICT-FLOOD-AI 코드와의 최종 호환성 분석 결과 출력
    status, summary, blockers, warnings, notices = analyze(result)
    print("\n[확인 필요 사항]")
    for item in blockers:
        print("[오류] " + item)
    for item in warnings:
        print("[주의] " + item)
    for item in notices:
        print("[안내] " + item)
    if not blockers and not warnings and not notices:
        print("없음")

    print("\n▶ 결 과 : [%s]" % status)
    if status == "사용 가능":
        print("  → 환경설정의 Python 경로로 사용할 수 있습니다.")
    elif status == "설치 후 확인 필요":
        print("  → 프로그램 실행 시 필요한 패키지가 설치됩니다.")
        print("  → 설치 완료 후 다시 검사하면 최종 호환성을 확인할 수 있습니다.")
    else:
        print("  → 현재 프로그램 코드와 호환되지 않아 사용을 권장하지 않습니다.")
    return status


# 전체 Python 환경 검사 시작
def main():
    print("=" * 78)
    print("KICT-FLOOD-AI Python 환경 검사")
    print("=" * 78)
    print("PC에 설치된 Python 중 KICT-FLOOD-AI에서 사용할 수 있는 환경을 확인합니다.")
    print("※ 검사만 수행하며 Python이나 패키지는 변경하지 않습니다.")
    print("※ 일반 Python과 venv 가상환경(pyvenv.cfg)을 함께 검색합니다.")

    # PC에서 검사 가능한 Python 실행 파일 검색
    pythons = discover_pythons()
    if not pythons:
        print("\n검사할 Python 실행 파일을 찾지 못했습니다.")
        return 1

    print("\n발견된 Python : %d개" % len(pythons))
    results = []
    # 발견된 Python 경로 출력
    for index, executable in enumerate(pythons, 1):
        print("  %d. %s" % (index, executable))

    # 각 Python 환경을 하나씩 독립적으로 검사
    for index, executable in enumerate(pythons, 1):
        print("\n(%d/%d) 검사 중..." % (index, len(pythons)))
        result = probe_python(executable)
        status = print_environment(index, len(pythons), result)
        results.append((result, status))

    # 검사한 전체 Python 환경을 한 번에 비교하여 출력
    print("\n" + "=" * 78)
    print("검사 결과 요약")
    print("=" * 78)
    print("%-10s %-18s %-14s %-20s" % ("Python", "TensorFlow", "NumPy", "결 과"))
    print("-" * 78)
    for result, status in results:
        packages = result.get("packages", {})
        print("%-10s %-18s %-14s %-20s" % (
            result.get("python_version", "?"),
            packages.get("TensorFlow") or "-",
            packages.get("NumPy") or "-",
            status,
        ))

    # 현재 코드 기준으로 사용 가능한 Python 환경 안내
    usable = [r for r, status in results if status == "사용 가능"]
    print("\n[사용 가능한 Python]")
    if usable:
        for result in usable:
            print("  Python %s" % result.get("python_version"))
            print("  경로 : %s" % result.get("executable"))
            print("  → 환경설정의 Python 경로로 지정할 수 있습니다.")
    else:
        print("  사용 가능 판정을 받은 Python이 없습니다.")
        print("  오류/주의 내용을 확인한 후 Python 환경을 다시 검사해주세요.")

    print("\n※ 모델 제작과 예측 수행에는 같은 Python 환경을 사용하는 것을 권장합니다.")
    return 0


# 이 파일을 직접 실행한 경우 전체 환경 검사 수행
if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n검사가 취소되었습니다.")
        exit_code = 130
    except Exception as exc:
        print("\n전체 환경 분석 중 오류: %s: %s" % (type(exc).__name__, str(exc)))
        exit_code = 1

    # Windows에서 결과 확인 전에 CMD 창이 바로 닫히지 않도록 대기
    if os.name == "nt":
        try:
            input("\n계속하려면 Enter 키를 누르세요...")
        except Exception:
            pass
    sys.exit(exit_code)
