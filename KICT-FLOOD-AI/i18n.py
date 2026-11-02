# i18n.py

TEXT = {
    "ko": {

        "waterlevel":"수위",
        "rainfall":"강우",
        "discharge":"유량",
        "daminlet":"댐 유입량",
        "damrelease":"댐 방류량",
        "tidelevel":"조위",
        "watershed":"유역평균강우",
        "predict_rain":"예측강우",

        "btn_pc_graph_save" : "그래프 저장",

        "msg_select_save_destination": "저장옵션(파일 또는 DB)을 선택하세요.",

        "label": "사용자 :",
        "label_67": "암 호 :",
        "label_72": "포 트 :",
        "label_68": "호스트/IP :",
        
        "btn_showPythonConsole": "파이썬콘솔창 보기",
        "btn_showDashboard": "대시보드",
        "btn_dialogClose": "프로그램 종료",

        "label_python_path": "Python 경로 :",
        "btn_set_python_path": "Python 경로 설정",
        #"label_model_format_file_path": "모델 파일 경로 :",
        "btn_set_model_path": "모델 경로 설정",
        "msg_default_lstm_cannot_be_deleted": "LSTM은 프로그램의 기본 모델이므로 삭제할 수 없습니다.\n"
                "Python 경로는 수정할 수 있습니다.",                

        "msg_f_model_algorithm_path_not_set": "{0} 모델의 실행 파일 경로가 설정되어 있지 않습니다.\n\n"
             "환경설정 → 모델 경로 설정에서 모델 경로를 확인하세요.",
        "msg_f_model_algorithm_folder_not_found": "{0} 모델의 프로그램 폴더를 찾을 수 없습니다.\n\n"
                "경로: {1}\n\n"
                "프로그램 설치 폴더의 ai_algorithm\\lstm 폴더가 "
                "삭제되거나 이동되지 않았는지 확인하세요.\n"
                "필요한 경우 프로그램을 다시 설치하세요.",
        "msg_f_model_execution_file_not_found": "{0} 모델을 실행하는 데 필요한 파일을 찾을 수 없습니다.\n\n"
                "필요한 파일: {1}\n"
                "확인한 경로: {2}\n\n"
                "ai_algorithm\\lstm 폴더의 프로그램 파일이 "
                "삭제되거나 누락되었는지 확인하세요.\n"
                "필요한 경우 프로그램을 다시 설치하세요.",

        "status_stop_requested": "중지 요청 중...",

        "label_hour_interval": "시간 간격",
        "label_minute_interval": "분 간격",

        "filedlg_open_modelparam_file": "ModelParam 파일 열기",
        "filedlg_save_file": "파일 저장",
        "filedlg_select_model_folder": "모델 폴더 선택",

        "btn_start_realtime_prediction": "실시간 예측 시작",
        "btn_stop_realtime_prediction": "실시간 예측 중지",

        "title_dataset_graph": "데이터셋 그래프",
        "btn_save_html": "HTML 저장",

        "btn_create": "생성",
        "btn_cancel": "취소",

        "label_layer_path": "레이어 경로 :",
        "label_station_code": "관측소 코드 :",
        "table_field_name": "필드명",
        "table_data_type": "데이터 형식",
        "table_sample_data": "샘플 데이터",
        "msg_failed_to_load_layer": "레이어를 불러오지 못했습니다.",

        "table_name": "테이블명",
        "table_sample_obs_id": "샘플 데이터(OBS_ID)",
        "table_sample_datetime": "샘플 데이터(DT_DATE)",
        "table_sample_original": "샘플 데이터(DT_DATA)",
        "table_sample_missing": "샘플 데이터(MI_DATA)",
        "table_sample_outlier": "샘플 데이터(OI_DATA)",
        "common_no_data": "데이터 없음",

        "table_file_name": "파일명",
        "table_file_path": "파일 경로",

        "title_bayesian_optimization": "베이지안 최적화",
        "group_parameter_setting": "파라미터 설정",
        "group_search_range": "탐색 범위",
        "label_min": "최솟값",
        "label_max": "최댓값",
        "btn_optimize_parameters": "파라미터 최적화",

        "title_bayesian_optimization_gru": "베이지안 최적화 - GRU",
        "title_bayesian_optimization_transformer": "베이지안 최적화 - Transformer",
        "label_bayesian_epoch": "베이지안 반복 횟수",
        "label_testing_epoch": "테스트 학습 횟수",
        "label_sequence_length": "시퀀스 길이",
        "label_dropout_rate": "드롭아웃 비율",
        "label_hidden_neuron": "은닉 뉴런 수",
        "label_number_of_layers": "레이어 수",
        "label_batch_size": "배치 크기",
        "label_gru_units": "GRU 유닛 수",
        "label_num_heads": "어텐션 헤드 수",
        "label_model_dimension": "모델 차원",
        "label_num_layers": "레이어 수",

        "table_hydro_model": "수문 모형",
        "table_rating_curve": "수위-유량 곡선",
        "table_basic_model": "기본 모형",
        "table_predict_rainfall_model": "예측강우 모형",
        "table_scenario_model": "시나리오 모형",

        "graph_water_level": "수위",
        "graph_flow_rate": "유량",
        "graph_observed_waterlevel": "관측 수위",
        "graph_predicted_waterlevel": "예측 수위",
        "graph_waterlevel_prediction_nse": "수위 예측 (NSE={})",
        "graph_flood_watch": "홍수주의보",
        "graph_flood_warning": "홍수경보",
        "graph_current_time": "현재 시각 :",

        "common_please_select": "선택해주세요",

        "title_model_path_set": "모델 경로 설정",
        "title_model_add": "모델 추가하기",
        "title_model_del": "모델 삭제하기",
        "label_model_name": "모델명 :",
        #"label_python_path": "Python 경로 :",
        "label_model_format_file_path": "모델 파일 경로 :",

        "btn_select_python": "경로 선택",
        "btn_select_folder": "폴더 선택",
        "btn_ok": "확인",
        "btn_cancel": "취소",

        "combo_item_hours": "시간(h)",
        "combo_item_minutes": "분(m)",

        "title_warning": "알림",
        "msg_enter_model_name": "모델명을 입력하세요.",
        "msg_select_python_path": "Python 경로를 선택하세요.",
        "msg_invalid_python_path": "선택한 Python 경로가 올바르지 않습니다.",
        "msg_select_model_format_file_path": "모델 형식 파일 경로를 선택하세요.",
        "msg_invalid_model_format_file_path": "선택한 모델 형식 파일 경로가 올바르지 않습니다.",
        
        "msg_closeEvent": "프로그램을 종료하시겠습니까?",
        "msg_env_enter_model_name": "모델명을 입력해주세요.",
        "msg_env_already_model_name": "동일한 모델명이 존재합니다. 다시 설정해주세요.",
        "msg_env_delete_model": "{} 모델을 삭제하시겠습니까?",

        "combo_item_normalized_data": "표준화자료",
        "combo_item_missing_value_correction": "결측치보정자료",
        "combo_item_outlier_filtered_data": "이상치보정자료",

        "placeholder_enter_station_id": "관측소ID 입력",
        "placeholder_search_field": "입력",

        "msg_data_count_differs_by_station": "관측소별로 데이터 개수가 다릅니다.",
        "msg_failed_to_get_data": "{data} 데이터를 가져오지 못했습니다.",
        "msg_target_leadtime_count_mismatch": "Target 관측소의 데이터 개수와 선행시간 데이터 개수가 일치하지 않습니다.",
        "msg_target_file_exists": "[Target 지점 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_target_file_not_exists": "[Target 지점 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_reference_file_exists": "[Reference 지점 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_reference_file_not_exists": "[Reference 지점 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_nse_file_exists": "[NSE 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_nse_file_not_exists": "[NSE 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_validation_file_exists": "[Validation 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_validation_file_not_exists": "[Validation 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_inference_file_exists": "[Inference 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_inference_file_not_exists": "[Inference 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_dataset_file_exists": "[{year}년 데이터셋 파일 정보]\n{path} 파일이 존재합니다.",
        "msg_dataset_file_not_exists": "[{year}년 데이터셋 파일 정보]\n{path} 파일이 존재하지 않습니다.",
        "msg_failed_to_process": "{obs_id} 처리에 실패했습니다. {error}",
        "msg_processing_error": "처리 중 오류가 발생했습니다.",
        "msg_database_save_error": "{obs_id} 데이터베이스 저장 오류: {error}",
        "msg_layer_not_found": "'{layer_name}' 레이어를 찾을 수 없습니다.",
        "msg_layer_loaded": "'{layer_name}' 레이어를 성공적으로 불러왔습니다.",
        "msg_no_available_tablespace": "[데이터베이스 설정] 사용 가능한 테이블스페이스가 없습니다.",
        "filedlg_select_python_executable": "Python 실행 파일 선택",
        "filedlg_select_model_format_folder": "모델 형식 폴더 선택",
        "msg_change_service_start_type": "[안내] {service_name} 서비스의 시작 유형을 '사용 안 함'에서 '수동'으로 변경합니다.",
        "msg_service_start_failed": "[오류] {service_name} 서비스를 시작하지 못했습니다.",
        "msg_failed_to_load_tablespaces": "[데이터베이스 설정] 테이블스페이스를 불러오지 못했습니다: {error}",
        "filedlg_open_file": "파일 열기",
        "filedlg_save_file": "파일 저장",
        "filedlg_save_html_file": "HTML 파일 저장",
        "msg_layer_connection_already_disconnected": "[데이터베이스 조회] 레이어 연결이 이미 해제되었습니다.",
        "msg_openapi_layer_connection_already_disconnected": "[OpenAPI 조회] 레이어 연결이 이미 해제되었습니다.",
        "msg_confirm_import_file": "선택한 파일을 {table_name} 테이블에 임포트 하시겠습니까?",
        "msg_query_database_error": "[데이터베이스 조회] 오류가 발생했습니다: {error}",
        "msg_query_database_exception": "[데이터베이스 조회] {error}",
        "error_load_python_console": "[모델 불러오기] Python 콘솔을 불러오지 못했습니다: {error}",
        "error_init_database_date": "[데이터베이스 설정] 데이터베이스 날짜 범위를 초기화하지 못했습니다: {error}",
        "error_load_shapefiles": "[레이어 설정] Shapefile을 불러오지 못했습니다: {error}",
        "error_terminate_db_import_thread": "[데이터베이스 가져오기] 가져오기 스레드를 종료하지 못했습니다: {error}",
        "error_wait_dataset_threadpool": "[데이터셋] 스레드 풀의 작업 완료를 기다리는 중 오류가 발생했습니다: {error}",

        "summary_target_point": "Target 지점",
        "summary_reference_point": "Reference 지점",
        "summary_period": "조회 기간",
        "summary_lead_time": "리드 타임",
        "summary_preprocessing_options": "전처리 옵션",

        "error_empty_target_point": "대상 지점 정보가 비어 있습니다.",
        "error_empty_lead_time": "예측 시간 정보가 비어 있습니다.",
        "error_invalid_query_period": "조회 기간 정보가 올바르지 않습니다.",
        "error_no_target_waterlevel_data": "선택한 조회 기간에 대상 지점의 수위 자료가 없습니다.",
        "error_insufficient_target_waterlevel_data": "선택한 예측 시간에 필요한 대상 지점의 수위 자료가 부족합니다.\n종료일을 앞당기거나 최대 예측 시간을 줄여 주세요.",
        "error_dataset_precheck_database": "데이터셋 사전 검사 중 데이터베이스 오류가 발생했습니다.\n{error}",
        "error_dataset_precheck": "데이터셋 사전 검사 중 오류가 발생했습니다.\n{error}",

        #"error_empty_lead_time": "[데이터셋 생성] 예측 시간 정보가 비어 있습니다.",
        "error_remove_existing_file": "기존 파일을 삭제하지 못했습니다:\n{path}\n\n{error}",
        "error_model_data_file_not_found": "[모델 생성] 데이터 파일을 찾을 수 없습니다: {path}",
        "error_model_data_load_failed": "[모델 생성] 데이터를 불러오지 못했습니다: {error}",

        "error_prediction_file_not_found": "[예측 및 성능 분석] 파일을 찾을 수 없습니다: {model_name}",
        "error_prediction_file_read_failed": "[예측 및 성능 분석] 파일을 읽는 중 오류가 발생했습니다: {error}",

        "log_load_model": "[예측 및 성능 분석] 모델을 불러오는 중입니다...",
        "log_model_loaded_successfully": "[예측 및 성능 분석] 모델을 성공적으로 불러왔습니다.",
        "log_loaded_model_type": "[예측 및 성능 분석] 불러온 모델 유형: {model_type}",
        "log_model_type_not_detected": "[예측 및 성능 분석] 모델 유형을 확인할 수 없습니다.",
        "log_model_load_failed": "[예측 및 성능 분석] 모델을 불러오지 못했습니다.",
        
        "log_prediction_start": "[예측 및 성능 분석] 예측을 시작합니다...",
        "log_prediction_failed": "[예측 및 성능 분석] 예측에 실패했습니다.",
        "error_prediction_file_or_python_not_found": "[예측 및 성능 분석] 파일 경로가 잘못되었거나 Python 실행 파일을 찾을 수 없습니다.",
        "error_prediction_exception": "[예측 및 성능 분석] 오류가 발생했습니다: {error}",
        "log_prediction_output": "=== 예측 실행 로그 ===",
        "error_prediction_log_read_failed": "[예측 및 성능 분석] 예측 로그를 읽지 못했습니다: {error}",
        "error_prediction_no_test_data": "[예측 및 성능 분석] 테스트 자료가 없어 예측을 수행할 수 없습니다.",
        "log_prediction_completed": "[예측 및 성능 분석] 예측이 완료되었습니다.",
        "error_prediction_failed_exit_code": "[예측 및 성능 분석] 예측에 실패했습니다. 종료 코드: {exit_code}",

        "log_realtime_model_summary": "[실시간 예측] 모델 요약",
        "log_realtime_error_occurred": "[실시간 예측] 오류가 발생했습니다: {error}",
        "log_realtime_python_not_found": "[실시간 예측] 파일 경로가 올바르지 않거나 Python 실행 파일을 찾을 수 없습니다.",
        "log_realtime_exception_occurred": "[실시간 예측] 예외가 발생했습니다: {error}",

        "log_realtime_mariadb_error": "[실시간 예측] MariaDB 오류가 발생했습니다: {error}",
        "log_realtime_exception_occurred": "[실시간 예측] 예외가 발생했습니다: {error}",

        "log_realtime_start": "[실시간 예측] 실시간 예측을 시작합니다.",
        "log_realtime_model_summary": "[실시간 예측] 모델 요약",
        "log_realtime_finish": "[실시간 예측] 실시간 예측이 완료되었습니다.",
        "log_realtime_warning": "[실시간 예측] 경고가 발생했습니다: {error}",
        "log_realtime_error_occurred": "[실시간 예측] 오류가 발생했습니다: {error}",
        "log_realtime_python_not_found": "[실시간 예측] 파일 경로가 올바르지 않거나 Python 실행 파일을 찾을 수 없습니다.",
        "log_realtime_exception_occurred": "[실시간 예측] 예외가 발생했습니다: {error}",

        "label_dataset_graph_period_info": "데이터셋 그래프 기간 정보 ({0} ~ {1})",
        "log_dashboard_model_parameter_not_found": "[대시보드] {target_point} 모델의 매개변수 정보를 가져올 수 없습니다.",

        "log_combo_item_not_found": "[알림] '{text}' 항목이 콤보박스에 존재하지 않습니다.",
        "log_all_files_processed_successfully": "모든 파일이 성공적으로 처리되었습니다.",

        "log_predict_rain_column_count_mismatch": "예측 강우 자료({obs_id})의 열 개수가 일치하지 않습니다. 예상: {expected}, 실제: {actual}",
        "log_temp_table_creation_error": "{obs_id}의 임시 테이블 생성 중 오류가 발생했습니다: {error}",
        "log_load_data_error": "{obs_id}의 데이터 로드 중 오류가 발생했습니다: {error}",
        "log_database_merge_error": "{obs_id}의 데이터베이스 병합 중 오류가 발생했습니다: {error}",
        "log_unsupported_column_count": "{obs_id}의 열 개수({column_count})는 지원되지 않습니다.",

        "log_initialize_file_read_failed": "[파일 초기화] 파일을 읽지 못했습니다: {error}",
        "log_initialize_file_min_columns": "[파일 초기화] CSV 파일에는 최소 2개의 열이 있어야 합니다.",
        "error_fewer_than_two_columns": "열이 2개 미만입니다.",
        "log_threehr_interpolation_error": "[보간] 오류가 발생했습니다: {error}",

        #################################################

        'msg_bayesian_optimization_failed_exit_code': "파라메터 최적화에 실패했습니다.\n종료 코드: {}",
        "msg_optimal_parameters_not_applied_files_deleted": "파라메터 최적화 결과를 적용하지 않았습니다.",        
        "msg_optimal_result_creation_controls": "파라메터 최적화 분석 결과를 적용하시겠습니까?",
        "msg_optimal_result_complete": "파라메터 최적화 분석이 완료되었습니다.\n\n",
        "msg_optimization_files_already_exist_run_again": "기존 모델 최적화 결과 파일이 존재합니다.\n\n"
            "기존 결과를 삭제하고 최적화를 다시 수행하시겠습니까?",
        "msg_failed_delete_existing_optimization_files": "기존 모델 최적화 결과 파일을 삭제하지 못했습니다.\n\n{}",

        'msg_info': '정보',
        #'msg_tr_self_current_language_message_key': '{message}',
        'msg_warning': '경고',
        'msg_error': '오류',
        'title_app': 'KICT FLOOD AI',        
        'title_db_connection_error': 'Database (MariaDB) 연결 오류',
        'msg_administrator_privileges_required_start_mariadb_service_continue_without': 'MariaDB 서비스를 시작하려면 관리자 권한이 '
                                                                                        '필요합니다.\n'
                                                                                        '\n'
                                                                                        'Database를 연결하지 않고 계속 진행하시겠습니까?',
        'msg_run_program_without_mariadb_some_features_may_limited': 'MariaDB 없이 프로그램을 실행합니다.\n일부 기능이 제한될 수 있습니다.',
        'msg_qgis_will_closed_restart_administrator_mode': 'QGIS를 종료합니다. 관리자 모드로 다시 실행하세요.',
        'title_db_service_start_error': 'Database (MariaDB) 서비스 시작 오류',
        'msg_f_exception_occurred_while_starting_mariadb_service_e': 'MariaDB 서비스를 시작하는 중 예외가 발생했습니다:\n{}',
        'msg_check_database_connection_information_again': '데이터베이스 연결 정보를 다시 확인하세요.',
        'title_performance_check': '성능 검토',
        'msg_failed_load_file_format_e': '파일을 불러오지 못했습니다.\n{}',
        'msg_selected_file_empty': '선택한 파일이 비어 있습니다.',
        'msg_not_find_datetime_column': '날짜/시간 컬럼을 찾을 수 없습니다.',
        'msg_selected_file_format_different_only_prediction_result_csv': '선택한 파일 형식이 다릅니다.\n'
                                                                         '\n'
                                                                         '예측 결과 CSV 파일만 불러올 수 있습니다.',
        'msg_no_valid_datetime_rows_found': '유효한 날짜/시간 행을 찾을 수 없습니다.',
        'msg_no_valid_datetime_rows_found_after_filtering': '필터링 후 유효한 날짜/시간 행을 찾을 수 없습니다.',
        'msg_not_find_observed_water_level_column': '관측 수위 컬럼을 찾을 수 없습니다.',
        'title_extreme_flood_scenario_creation': '대형홍수시나리오 생성',
        'msg_there_no_information_selected_folder': '선택한 폴더에 정보가 없습니다.',
        'msg_enter_parent_folder_path_modelparam': 'ModelParam 상위 폴더 경로를 입력하세요.',
        'msg_enter_folder_path_extreme_flood_scenario_model': '대형홍수시나리오 모델 폴더 경로를 입력하세요.',
        'msg_enter_save_path_scenario_dataset': '시나리오 데이터셋 저장 경로를 입력하세요.',
        'msg_enter_folder_containing_observation_data_files': '관측 데이터 파일이 들어 있는 폴더를 입력하세요.',
        'msg_select_extreme_flood_scenario_name': '대형홍수시나리오명을 선택하세요.',
        'msg_cannot_extract_water_level_observation_station_code_scenario': '시나리오명에서 수위 관측소 코드를 추출할 수 없습니다.',
        'msg_f_cannot_load_scenario_module_make_scenario_import': '시나리오 모듈(.make_scenario)을 불러올 수 없습니다.\n{}',
        'msg_scenario_creation_complete_want_check_file': '시나리오 생성이 완료되었습니다. 파일을 확인하시겠습니까?',
        'msg_msg': '{msg}',
        'title_query_database': '데이터베이스 조회',
        'msg_select_type_table_query': '조회할 테이블 종류를 선택하세요.',
        'msg_war_msg': '{war_msg}',
        'msg_no_layer_selected': '선택된 레이어가 없습니다.',
        'title_query_openapi': 'OpenAPI 조회',
        'msg_select_type_data_query': '조회할 데이터 종류를 선택하세요.',
        'title_dashboard': '대시보드',
        'msg_set_folder_path_data_display_on_dashboard': '대시보드에 표시할 데이터 폴더 경로를 설정하세요.',
        'msg_no_folder_selected': '선택된 폴더가 없습니다.',
        'title_data_file_db_import': '데이터 파일 DB 가져오기',
        'msg_there_no_data_import': '가져올 데이터가 없습니다.',
        'msg_msgimport': '{msgImport}',
        'msg_operation_been_canceled': '작업이 취소되었습니다.',
        'msg_if_duplicate_data_exists_want_update': '중복 데이터가 있으면 업데이트하시겠습니까?',
        'msg_e': '{e}',
        'msg_all_files_been_saved': '모든 파일이 저장되었습니다.',
        'msg_invalid_internet_connection': '인터넷 연결이 올바르지 않습니다.',
        'msg_data_result_errdata': '{error}',
        'msg_data_errdata': '{error}',
        'msg_errmsg': '{errMsg}',
        'msg_enter_observation_station_number_query': '조회할 관측소 번호를 입력하세요.',
        'msg_start_date_cannot_later_than_end_date': '시작일은 종료일보다 늦을 수 없습니다.',
        'msg_query_period_limited_one_day': '조회 기간은 1일로 제한됩니다.',
        'msg_query_period_limited_one_month': '조회 기간은 1개월로 제한됩니다.',
        'msg_there_no_data_save': '저장할 데이터가 없습니다.',
        'msg_data_been_saved_successfully': '데이터가 성공적으로 저장되었습니다.',
        'msg_select_table_query': '조회할 테이블을 선택하세요.',
        'msg_save_table_query': '저장할 테이블을 선택하세요.',
        'msg_enter_observation_station_id_query': '조회할 관측소 ID를 입력하세요.',
        'msg_no_data_found': '데이터가 없습니다.',
        'title_dataset_creation': '데이터셋 생성',
        'msg_select_only_one_target_point': 'Target Point는 하나만 선택하세요.',
        'msg_no_feature_selected': '선택된 피처가 없습니다.',
        'msg_enter_select_target_point_information_query': '조회할 Target Point 정보를 입력하거나 선택하세요.',
        'msg_enter_select_reference_point_information_query': '조회할 Reference Point 정보를 입력하거나 선택하세요.',
        'msg_target_point_also_included_reference_point_list_remove': 'Target Point가 Reference Point 목록에도 포함되어 있습니다.\n'
                                                                      '중복된 지점을 제거한 후 다시 시도하세요.',
        'msg_set_period_query': '조회 기간을 설정하세요.',
        'msg_no_rainfall_point_information_selected_so_dry_weather': '강우 지점 정보가 선택되지 않아 무강우 사상 데이터를 확인할 수 없습니다.',
        'msg_recheck_information_preprocessing_option_delete_dry_weather_event': '전처리 옵션 정보를 다시 확인하세요: 무강우 사상 삭제.',
        'msg_recheck_input_information_preprocessing_option_delete_dry_weather': '전처리 옵션 입력값을 다시 확인하세요: 무강우 사상 삭제.\n'
                                                                                 '(예: 양의 정수 입력)',
        'msg_enter_both_water_level_change_time_threshold_no': '무강우 사상 옵션의 수위변화 시간과 임계값을 모두 입력하세요.',
        'msg_enter_water_level_change_time_as_positive_integer': '수위변화 시간은 양의 정수로 입력하세요.',
        'msg_enter_water_level_change_threshold_as_number_greater': '수위변화 임계값은 0 이상의 숫자로 입력하세요.',
        'msg_no_rainfall_point_information_selected_so_cumulative_rainfall': '강우 지점 정보가 선택되지 않아 누적강우 데이터를 생성할 수 없습니다.',
        'msg_recheck_information_preprocessing_option_generate_cumulative_rainfall': '전처리 옵션 정보를 다시 확인하세요: 누적강우 생성.',
        'msg_recheck_input_information_preprocessing_option_generate_cumulative_rainfall': '전처리 옵션 입력값을 다시 확인하세요: 누적강우 '
                                                                                           '생성.\n'
                                                                                           '(예: 3시간, 6시간은 3,6 입력)\n'
                                                                                           '양의 정수만 허용되며, 중복값과 빈 값은 사용할 '
                                                                                           '수 없습니다.',
        'msg_recheck_information_preprocessing_option_period_filtering': '전처리 옵션 정보를 다시 확인하세요: 기간 필터링.',
        'msg_no_watershed_information_selected_so_predictive_rainfall_data': '유역 정보가 선택되지 않아 예측강우 데이터를 생성할 수 없습니다.',
        'msg_recheck_information_preprocessing_option_generate_predictive_rainfall': '전처리 옵션 정보를 다시 확인하세요: 예측강우 생성.',
        'msg_recheck_input_information_preprocessing_option_generate_predictive_rainfall': '전처리 옵션 입력값을 다시 확인하세요: 예측강우 '
                                                                                           '생성.\n'
                                                                                           '(예: 3시간, 6시간은 3,6 입력)\n'
                                                                                           '양의 정수만 허용되며, 중복값과 빈 값은 사용할 '
                                                                                           '수 없습니다.',
        'msg_set_lead_time_information_query': '조회할 Lead time 정보를 설정하세요.',
        'msg_recheck_time_series_lead_time_settings': '시계열 Lead time 설정을 다시 확인하세요.',
        'msg_lead_time_interval_cannot_greater_than_target_lead': 'Lead time 간격은 목표 Lead time보다 클 수 없습니다.',
        'msg_f_e': '{}',
        'msg_dataset_creation_been_completed': '데이터셋 생성이 완료되었습니다.',
        'msg_precheck_msg': '{precheck_msg}',
        'msg_f_create_dataset_configured_information_summary': '설정한 정보로 데이터셋을 생성하시겠습니까?\n{}',
        'msg_save_without_null_values_yes_save_excluding_null': 'NULL 값을 제외하고 저장하시겠습니까?\n'
                                                                '(Yes: NULL 값을 제외하고 저장, No: 전체 데이터 저장)',
        'msg_dataset_file_been_saved_want_check_file': '데이터셋 파일이 저장되었습니다. 파일을 확인하시겠습니까?',
        'msg_no_data_query': '조회할 데이터가 없습니다.',
        'title_model_creation': '모델 생성',
        'msg_f_check_contents_modelparameter_file_following_keys_missing': 'ModelParameter 파일 내용을 확인하세요.\n'
                                                                           "다음 key가 누락되었습니다: '{}'.",
        'msg_f_want_update_current_value_current_value_origin': '현재 값을 업데이트하시겠습니까?\n\n현재 값 : {}\n새 값 : {}',
        'msg_select_dataset_file': '데이터셋 파일을 선택하세요.',
        'msg_dataset_file_contains_null_values': '데이터셋 파일에 NULL 값이 포함되어 있습니다.',
        'msg_select_folder_save_model': '모델을 저장할 폴더를 선택하세요.',
        #'msg_enter_model_name': '모델명을 입력하세요.',
        'msg_select_model': '모델을 선택하세요.',
        'msg_python_executable_path_selected_model_empty': '선택한 모델의 Python 실행 파일 경로가 비어 있습니다.',
        'msg_model_folder_path_selected_model_empty': '선택한 모델의 모델 폴더 경로가 비어 있습니다.',
        'msg_check_hidden_layer_input_values_example_32_64': 'Hidden Layer 입력값을 확인하세요.\n(예: 32,64,32)',
        'msg_enter_values_according_number_hidden_layers': 'Hidden Layer 개수에 맞게 값을 입력하세요.',
        'msg_enter_values_matching_number_hidden_layers': 'Hidden Layer 개수와 일치하는 값을 입력하세요.',
        'msg_select_data_columns_use_model_creation': '모델 생성에 사용할 데이터 컬럼을 선택하세요.',
        'msg_select_lead_time_column_use_model_creation': '모델 생성에 사용할 Lead time 컬럼을 선택하세요.',
        'msg_enter_standard_watershed_code_information': '기준 유역 코드를 입력하세요.',
        'msg_standard_watershed_code_ws_information_missing_not_selected': '데이터셋에 기준 유역 코드(WS) 정보가 없거나 선택되지 않았습니다.',
        'msg_standard_watershed_code_predict_ws_information_applying_observed': '관측강우 적용을 위한 기준 유역 코드(Predict_WS) 정보가 '
                                                                                '데이터셋에 없거나 선택되지 않았습니다.',
        'msg_use_apply_realtime_predict_ws_set_observed_data': 'Apply Realtime Predict_WS를 사용하려면 시스템 환경설정 → 대시보드 설정에서 '
                                                               '관측 데이터 폴더 경로를 설정하세요.',
        'msg_f_no_test_data_set_want_proceed_anyway': 'Test 데이터가 설정되지 않았습니다. 그래도 계속하시겠습니까?',
        'msg_reset_test_data': 'Test 데이터를 다시 설정하세요.',
        'title_model_optimization': '모델 최적화',
        'msg_f_bayesian_py_file_not_found_bayesian_py': 'bayesian.py 파일을 찾을 수 없습니다.\n\n{}',
        'msg_python_executable_path_invalid': 'Python 실행 파일 경로가 올바르지 않습니다.',
        'msg_model_folder_path_invalid': '모델 폴더 경로가 올바르지 않습니다.',
        'msg_f_bayesian_py_not_found_script_path': 'bayesian.py를 찾을 수 없습니다.\n\n{}',
        'msg_f_failed_start_bayesian_optimization_e': 'Bayesian Optimization을 시작하지 못했습니다.\n\n{}',
        'msg_f_bayesian_optimization_failed_exit_code_exit_code': 'Bayesian Optimization에 실패했습니다.\n'
                                                                  'Exit code: {}\n'
                                                                  '\n'
                                                                  '로그 경로:\n'
                                                                  '{}\n'
                                                                  '\n'
                                                                  '마지막 로그:\n'
                                                                  '{}',
        'msg_f_best_params_csv_not_found_best_params': 'best_params.csv를 찾을 수 없습니다.\n\n{}',
        'msg_f_optimization_trials_csv_not_found_trials_path': 'optimization_trials.csv를 찾을 수 없습니다.\n\n{}',
        'msg_f_best_params_csv_empty_best_params_path': 'best_params.csv가 비어 있습니다.\n\n{}',
        'msg_summary': '{summary}',
        'msg_optimal_parameters_been_applied_model_creation_controls': '최적 파라미터가 모델 생성 컨트롤에 적용되었습니다.',
        'msg_f_optimal_parameters_saved_only_best_params_csv': '최적 파라미터가 저장만 되었습니다.\n'
                                                               '\n'
                                                               'best_params.csv:\n'
                                                               '{}\n'
                                                               '\n'
                                                               'optimization_trials.csv:\n'
                                                               '{}',
        'msg_f_failed_read_apply_best_params_csv_e': 'best_params.csv를 읽거나 적용하지 못했습니다.\n\n{}',
        'msg_folder_already_exists_want_overwrite': '폴더가 이미 존재합니다. 덮어쓰시겠습니까?',
        'msg_path_contains_korean_characters_change_path_use_only': '경로에 한글이 포함되어 있습니다. 영문 경로만 사용하도록 변경하세요.',
        'msg_f_watershed_rainfall_column_not_found_dataset_missing': '데이터셋에서 유역강우 컬럼을 찾을 수 없습니다.\n\n누락된 컬럼 : {}',
        'msg_f_makemodel_py_not_found_script_path': 'makemodel.py를 찾을 수 없습니다.\n\n{}',
        'msg_f_modelparam_txt_not_found_modelpath': 'modelParam.txt를 찾을 수 없습니다.\n\n{}',
        'msg_model_created_successfully_however_there_no_test_data': '모델이 성공적으로 생성되었습니다.\n'
                                                                     '\n'
                                                                     '하지만 Test 데이터가 없어 예측/평가 결과는 생성되지 않았습니다.\n'
                                                                     '\n'
                                                                     '결과 폴더를 확인하시겠습니까?',
        'msg_model_been_successfully_created_want_view_results': '모델 생성이 완료되었습니다. 결과를 확인하시겠습니까?',
        'msg_f_model_creation_failed_exit_code_exit_code': '모델 생성에 실패했습니다. Exit code: {}',
        'msg_dataset_file_does_not_contain_target_point_information': '데이터셋 파일에 Target point 정보가 없습니다.',
        'msg_dataset_file_does_not_contain_leadtime_information': '데이터셋 파일에 Leadtime 정보가 없습니다.',
        'msg_dataset_file_does_not_contain_reference_point_information': '데이터셋 파일에 Reference point 정보가 없습니다.',
        'title_prediction_performance_analysis': '예측 수행 및 성능 분석',
        'msg_select_model_folder_path': '모델 폴더 경로를 선택하세요.',
        'msg_select_folder_path_save_prediction_results': '예측 결과를 저장할 폴더 경로를 선택하세요.',
        'msg_enter_name_prediction_results': '예측 결과 이름을 입력하세요.',
        'msg_select_data_use_prediction': '예측에 사용할 데이터를 선택하세요.',
        'msg_number_data_items_used_model_data_prediction_not': '모델에 사용된 데이터 항목 수와 예측 데이터 항목 수가 일치하지 않습니다.',
        'msg_types_data_used_model_data_selected_prediction_not': '모델에 사용된 데이터 종류와 예측에 선택한 데이터 종류가 일치하지 않습니다.',
        'msg_select_lead_time_use_prediction': '예측에 사용할 Lead time을 선택하세요.',
        'msg_number_lead_times_used_model_lead_times_selected': '모델에 사용된 Lead time 수와 선택한 Lead time 수가 일치하지 않습니다.',
        'msg_types_lead_times_used_model_lead_times_selected': '모델에 사용된 Lead time 종류와 선택한 Lead time 종류가 일치하지 않습니다.',
        'msg_standard_watershed_codes_used_model_missing_not_selected': '모델에 사용된 기준 유역 코드가 데이터셋에 없거나 선택되지 않았습니다.',
        'msg_f_file_modelname_not_found': '{} 파일을 찾을 수 없습니다.',
        'msg_f_result_stderr': '{}',
        'msg_f_e_stderr': '{}',
        'msg_f_file_path_incorrect_python_executable_not_found': '파일 경로가 올바르지 않거나 Python 실행 파일을 찾을 수 없습니다.',
        'msg_f_failed_prepare_result_folder_e': '결과 폴더를 준비하지 못했습니다.\n\n{}',
        'msg_prediction_not_performed_because_there_no_test_data': '선택한 데이터셋에 Test 데이터가 없어 예측을 수행할 수 없습니다.',
        'msg_prediction_been_completed_want_view_results': '예측이 완료되었습니다. 결과를 확인하시겠습니까?',
        'msg_f_prediction_failed_exit_code_exit_code': '예측에 실패했습니다. Exit code: {}',
        'title_realtime_prediction': '실시간 예측',
        'msg_select_prediction_model_use_real_time_prediction': '실시간 예측에 사용할 예측 모델을 선택하세요.',
        'msg_select_folder_path_save_real_time_prediction_results': '실시간 예측 결과를 저장할 폴더 경로를 선택하세요.',
        'msg_number_data_items_varies_by_observation_station': '관측소별 데이터 항목 수가 다릅니다.',
        'msg_sure_want_exit': '그래프 화면을 종료하시겠습니까?',
        'title_dataset_graph_save': '데이터셋 그래프 저장',
        'msg_there_no_graph_html_file_save': '저장할 그래프 HTML 파일이 없습니다.',
        'msg_f_html_saved_file_path': 'HTML이 저장되었습니다:\n{}',
        'msg_f_failed_save_html_file_e': 'HTML 파일 저장에 실패했습니다.\n{}',
        'msg_there_no_graph_save': '저장할 그래프가 없습니다.',
        'title_dashboard_file_information': '대시보드 파일 정보',
        'title_dashboard_report': '대시보드 보고서',
        'msg_no_format_file_found_report_creation': '보고서 생성을 위한 양식 파일을 찾을 수 없습니다.',
        'msg_failed_open_hwp_check_whether_hancom_office_hwp': 'HWP를 열지 못했습니다. 한컴오피스 HWP가 설치되어 있는지 또는 파일을 확인하세요.',
        'title_create_database': '데이터베이스 생성',
        'info_create_database': '데이터베이스 이름을 입력하세요.',
        'msg_enter_name': '이름을 입력하세요.',        
        'msg_check_predictrain_table': '예측강우테이블(predict_rain) 생성',
        'msg_input_can_contain_only_letters_numbers_underscores': '(*공백없이 영문,숫자,밑줄(_)로만 입력 가능)',
        'msg_f_name_database_already_exists': "'{}' 데이터베이스가 이미 존재합니다.",
        'msg_f_error_occurred_e': '오류가 발생했습니다.\n{}',
        #'title_bayesian_optimization': 'Bayesian Optimization',
        'msg_f_label_text_min_value_cannot_greater_than': '{} 최소값은 최대값보다 클 수 없습니다.',
        'title_connect_database': '데이터베이스 연결 설정',
        'info_connect_database': '데이터베이스 연결 정보를 설정합니다.',
        'btn_fileselectdlg_apply': '적용하기',
        'btn_fileselectdlg_cancel': '취소',
        'btn_fileselectdlg_selFile': '파일선택',

        "msg_input_error_title": "입력 오류",
        "msg_fileselectdlg_select_layer_file": "레이어 파일을 선택하세요.",
        "msg_fileselectdlg_layer_file_not_exist": "선택한 레이어 파일이 존재하지 않습니다.",
        "msg_fileselectdlg_invalid_shapefile": "올바른 Shapefile(*.shp)을 선택하세요.",
        "msg_fileselectdlg_select_station_code": "관측소 코드 필드를 선택하세요.",
        "msg_fileselectdlg_field_info_not_loaded": "레이어 필드 정보를 불러오지 못했습니다.",
        "msg_fileselectdlg_failed_load_layer": "선택한 레이어를 불러오지 못했습니다.",
        "msg_fileselectdlg_station_field_not_exist": "선택한 관측소 코드 필드가 레이어에 존재하지 않습니다.",

        #################################################

        # QTabWidget
        "tab_dataset": "데이터셋 생성",
        "tab_model": "하천수위 모형제작",
        "tab_predict": "예측수행 및 성능분석",
        "tab_realtime": "실시간 예측",
        "tab_performanceCheck": "성능 분석",
        "tab_dataMng": "데이터 관리",
        "tab_makrScenario": "대형홍수시나리오 생성",
        "tab_envSet": "환경설정",
        
        "tab_ds_original": "데이터셋 원본",
        "tab_ds_preprocessing": "데이터셋 전처리",
        
        "tab_3": "데이터파일 DB Import",
        "tab_5": "DB 자료조회",
        "tab_7": "OpenAPI 자료조회",
        
        "tab_6": "수문학적모형 (저류함수법)",
        
        # QToolbox
        "tb_set_point": "1단계 : 지점선택",
        "tb_set_period": "2단계 : 조회기간 및 리드타임 선택",
        "tb_set_option": "옵션 : 데이터 전처리",
        
        # QPushButton
        "btn_dataset_activelayer_targetpoint": "선택",
        "btn_dataset_activelayer_waterlevel": "선택",
        "btn_dataset_activelayer_rainfall": "선택",
        "btn_dataset_activelayer_discharge": "선택",
        "btn_dataset_activelayer_daminlet": "선택",
        "btn_dataset_activelayer_damrelease": "선택",
        "btn_dataset_activelayer_tidelevel": "선택",
        "btn_dataset_activelayer_watershed": "선택",
        "btn_dataset_selLayer_init": "초기화",
        
        "btn_makeDataset": "데이터셋 생성",
        "btn_ds_origin_prev": "이전페이지",
        "btn_ds_origin_next": "다음페이지",
        "btn_ds_preset_prev": "이전페이지",
        "btn_ds_preset_next": "다음페이지",
        "btn_showDatasetGraph": "그래프보기",
        "btn_saveDataset": "CSV파일저장",
        
        "btn_param_datasetPath": "선택",
        "btn_param_savePath": "선택",
        "btn_dl_dataset_prev": "이전페이지",
        "btn_dl_dataset_next": "다음페이지",
        "btn_makemodel_optimize": "모델 최적화",
        "btn_dl_loadParameter": "모델파라메터파일 불러오기",
        "btn_makeModel": "하천수위 모형제작",
        
        "btn_selectDataFile": "선택",
        "btn_selectModelFile": "선택",
        "btn_load_dl_dataset_prev": "이전페이지",
        "btn_load_dl_dataset_next": "다음페이지",
        "btn_dl_reloadModel": "예측 수행",
        
        "btn_selectRealtimeModelFile": "모델 선택",
        "btn_saveRealtimeModelPath": "선택",
        "btn_realtimeModel": "실시간예측  On",
        
        "btn_selectPerformResultFile": "선택",
        "btn_perform_createWindow": "윈도우 생성",
        
        "btn_data_loadDataFile": "선택",
        "btn_dataSave": "저장하기",
        "btn_datasearchGispoint": "지도에서 지점선택",
        "btn_dataSearch": "조 회",
        "btn_apisearchGispoint": "지도에서 지점선택",
        "btn_apiSearch": "조 회",
        "btn_apiSaveCSV": "파일저장",
        
        "btn_sceModel_select_scenario_model_folder": "폴더설정하기",
        "btn_sceModel_select_modelParam_folder": "폴더설정하기",
        "btn_sceModel_select_dataset_folder": "폴더설정하기",
        "btn_sceModel_select_observ_folderpath": "폴더설정하기",
        "btn_makeScenario_SFM": "대형홍수 시나리오 생성",
        
        "btn_envSet_database": "데이터베이스 설정",
        "btn_envSet_layerinfo": "레이어 설정",
        "btn_envSet_dashboard": "대시보드 설정",
        "btn_envSet_python": "모델경로 설정",

        "btn_connect_mariadb": "데이터베이스 연결정보 설정",
        "btn_create_tablespace": "데이터베이스 새로 생성하기",
        
        "btn_env_lyr_waterlevel": "변경하기",
        "btn_env_lyr_rainfall": "변경하기",
        "btn_env_lyr_discharge": "변경하기",
        "btn_env_lyr_daminlet": "변경하기",
        "btn_env_lyr_damrelease": "변경하기",
        "btn_env_lyr_tidelevel": "변경하기",
        "btn_env_lyr_watershed": "변경하기",
        "btn_env_lyr_rivernetwork": "변경하기",
        "btn_env_lyr_default": "기본레이어로 설정하기",
        
        "btn_dash_modify_model": "수정",
        "btn_dash_modify_year": "수정",
        "btn_dash_modify_riverRegion": "수정",
        "btn_dash_modify_modelName": "수정",
        
        "btn_config_model_add": "추가",
        "btn_config_model_del": "삭제",
        "btn_check_pythonPath": "파이썬 경로확인",
        
        # QGroupBox        
        "gb_datasetLayer": "데이터셋 생성정보",
        "groupBox_8": "조회기간 설정",
        "groupBox_9": "리드타임 설정",
        "groupBox_13": "무강우 사상 제외",
        "groupBox_20": "누적강우 생성",
        "groupBox_10": "예측강우 생성",

        "groupBox_makemodel_dataset": "데이터셋 파일 정보 설정",
        "groupBox_makemodel_param": "파라메터 정보 설정",
        "groupBox_makemodel_save": "저장경로 설정",
        
        "groupBox_predict_dataset": "데이터셋 파일 정보 설정",
        "groupBox_predict_model": "모델 선택",
        "groupBox_predict_dl": "데이터셋 및 리드타임 정보 설정",
        "groupBox_predict_save": "저장경로 설정",
        
        "groupBox_31": "실시간 하천수위예측 수행 정보 설정",
        
        "groupBox": "파일 설정",
        
        "groupBox_18": "데이터베이스 설정",
        "groupBox_30": "MariaDB 연결 정보를 설정합니다.",
        "groupBox_25": "데이터베이스 정보를 설정합니다.",
        "groupBox_19": "레이어 설정",
        "groupBox_22": "대시보드 설정",
        "groupBox_26": "모델이 저장되어 있는 상위폴더 정보를 설정합니다.",
        "groupBox_27": "각 레벨별 항목정보를 추가,수정,삭제합니다.",
        "groupBox_28": "관측자료 정보가 저장되어 있는 폴더를 설정합니다.",
        "groupBox_29": "모델별 파이썬 실행파일 및 모델파일 경로 설정",   
        
        # QChceckBox
        "ckb_ds_preProcessing_noRain": "무강우사상 삭제 - 전후",
        "ckb_ds_preProcessing_checkWL": "수위옵션 설정",
        "sp_ds_preProcessing_noRainVal": "누적강우 생성",
        "ckb_ds_preProcessing_month": "기간 필터링",
        "ckb_ds_preProcessing_predictRainVal": "예측강우 생성",
        
        "ckb_data_option2": "결측치보정",
        "ckb_data_option1": "이상치보정",
        "label_8": "보간구간(H)",
        "ckb_data_saveFile": "CSV 파일",
        "ckb_data_saveDbImport": "데이터베이스",
        
        # QRadioButton
        "rb_dataset_periodDate": "상세기간별 조회하기",
        "rb_dataset_periodYear": "연도별 조회하기",
        "rb_ds_leadtimeBasic": "리드타임 선택",
        "rb_ds_leadtimeTimeseries": "Timeseries 설정",
        "rb_ds_preProcessing_month": "특정월 조회",
        "rb_ds_preProcessing_date": "특정기간 조회",
        
        "rb_perform_manual": "수동설정 ",
        "rb_perform_automatic": "자동탐색",
        "perform_option_peak_waterlevel": "최대수위기준",
        "perform_option_peak_rainfall": "최대강우기준",
        
        "rb_observ_folderpath_db": "데이터베이스",
        "rb_observ_folderpath_file": "파일",
        
        # QLabel
        "lbl_env_waterlevel_2": "데이터베이스 설정 :",
        "label_93": "폴더경로 :",
        "label_97": "[참고] Level4(Target Point)의 정보는 Level3에서 선택된 권역 내 폴더정보를 이용해 가져옵니다.",
        "label_99": "폴더경로 :",
        
        "label_14": """
        <html>
        <head>
        <style type="text/css">
        p {
            margin-top: 0px;
            margin-bottom: 4px;
            margin-left: 0px;
            margin-right: 0px;
            white-space: pre-wrap;
        }
        </style>
        </head>
        <body style="font-family:'Gulim'; font-size:9pt; font-weight:400; font-style:normal;">
        <p>※ <span style="font-weight:600;">Check Python Path</span> 버튼을 클릭한 후, 팝업 창에서 <span style="font-family:'Courier New';">checkpythonpath.py</span> 파일을 선택하고 <span style="font-weight:600;">Python 연결 프로그램</span>으로 실행하세요.</p>
        <p>※ 명령 프롬프트(cmd) 창에 표시되는 Python 실행 파일 경로를 확인해주세요.</p>
        <p>※ Python이 설치되어 있지 않으면 연결 프로그램 옵션이 표시되지 않습니다.</p>
        <p>※ <span style="font-weight:600;">하천 수위 예측 모델 개발</span>을 수행하려면 Python을 다운로드하여 설치해야 합니다.</p>
        </body>
        </html>
        """,
        
        "label_84": "조회년도 :",
        "label_83": "목표시간 :",
        "label_88": "시간간격 :",
        
        "label_92": "실시간예측 시작시간 설정 :",
        
        "label_85": "시작날짜 :",
        "label_86": "기간 :",
        "label_102": "범위 :",
        
        "label_78": "데이터 파일선택 :",
        "label_82": "설정 :",
        "label_81": "테이블 선택",
        "label_80": "데이터 품질관리(옵션)",
        "label_100": "저장옵션",

        "label0_2": "조회테이블 :",
        "label01_3": "관측소번호 :",
        "label01_4": "조회기간 :",

        "label_60": """
            <html><head/><body>
            <p>한강홍수통제소자료(수위,유량,댐유입/방류량-10분단위,최대1개월), 국립해양조사원(조위-1분단위,최대1일) 자료를 OpenAPI를 통해 가져옵니다.</p>
            </body></html>
            """,  
        "label_2": "자료종류 :",
        "label_3": "관측소코드 :",    
        "label_4": "조회기간 :",  
        "label_23": "조회결과를 CSV파일로 저장합니다. 관측소코드가 기본파일명으로 저장됩니다.",

        
        "label_130": "대형홍수시나리오모델 폴더경로 :",
        "label_133": "대형홍수시나리오명 :",
        "label_128": "ModelParam 상위폴더경로 :",
        "label_129": "권역명 :",
        "label_131": "데이터셋 저장경로 :",
        "label_134": "관측데이터 적용방법 :",

    },

    "en": {
        
        "waterlevel":"WaterLevel",
        "rainfall":"Rainfall",
        "discharge":"Discharge",
        "daminlet":"DamInflow",
        "damrelease":"DamOutflow",
        "tidelevel":"TideLevel",
        "watershed":"WatershedAverageRainfall",
        "predict_rain":"PredictedRainfall",
        
        "btn_pc_graph_save" : "Save Graph",

        "msg_select_save_destination": "Please select a save option (File or Database).",
        
        "label": "User :",
        "label_67": "Password :",
        "label_72": "Port :",
        "label_68": "Host/IP :",

        "btn_showPythonConsole": "Show PyhonConsole",
        "btn_showDashboard": "Dashboard",
        "btn_dialogClose": "Close",

        "label_python_path": "Python Path :",
        "btn_set_python_path": "Set Python Path",
        #"label_model_format_file_path": "Model Format File Path :",
        "btn_set_model_path": "Set Model Path",        
        "msg_default_lstm_cannot_be_deleted": "LSTM is the program's default model and cannot be deleted.\n"
            "You can modify its Python path.",

        "msg_f_model_algorithm_path_not_set": "The execution path for the {0} model has not been set.\n\n"
                "Check the model path in Settings → Model Path Settings.",
        "msg_f_model_algorithm_folder_not_found": "The program folder for the {0} model could not be found.\n\n"
                "Path: {1}\n\n"
                "Check whether the ai_algorithm\\lstm folder in the "
                "program installation directory has been deleted or moved.\n"
                "Reinstall the program if necessary.",
        "msg_f_model_execution_file_not_found": "A file required to run the {0} model could not be found.\n\n"
                "Required file: {1}\n"
                "Checked path: {2}\n\n"
                "Check whether files in the ai_algorithm\\lstm folder "
                "have been deleted or are missing.\n"
                "Reinstall the program if necessary.",

        "status_stop_requested": "Stop Requested...",

        "label_hour_interval": "hour interval",
        "label_minute_interval": "minute interval",

        "filedlg_open_modelparam_file": "Open ModelParam File",
        "filedlg_save_file": "Save File",
        "filedlg_select_model_folder": "Select Model Folder",

        "btn_start_realtime_prediction": "Start Real-Time Prediction",
        "btn_stop_realtime_prediction": "Stop Real-Time Prediction",

        "title_dataset_graph": "Dataset Graph",
        "btn_save_html": "Save HTML",

        "btn_create": "Create",
        "btn_cancel": "Cancel",
        
        "combo_item_hours": "Hours",
        "combo_item_minutes": "Minutes",

        "label_layer_path": "Layer Path :",
        "label_station_code": "Station Code :",
        "table_field_name": "Field Name",
        "table_data_type": "Data Type",
        "table_sample_data": "Sample Data",
        "msg_failed_to_load_layer": "Failed to load layer.",

        "table_name": "Table Name",
        "table_sample_obs_id": "Sample Data(OBS_ID)",
        "table_sample_datetime": "Sample Data(DT_DATE)",
        "table_sample_original": "Sample Data(DT_DATA)",
        "table_sample_missing": "Sample Data(MI_DATA)",
        "table_sample_outlier": "Sample Data(OI_DATA)",
        "common_no_data": "No Data",

        "table_file_name": "File Name",
        "table_file_path": "File Path",

        "title_bayesian_optimization": "Bayesian Optimization",
        "group_parameter_setting": "Parameter Setting",
        "group_search_range": "Search Range",
        "label_min": "Min",
        "label_max": "Max",
        "btn_optimize_parameters": "Optimize Parameters",

        "title_bayesian_optimization_gru": "Bayesian Optimization - GRU",
        "title_bayesian_optimization_transformer": "Bayesian Optimization - Transformer",
        "label_bayesian_epoch": "Bayesian Epoch",
        "label_testing_epoch": "Testing Epoch",
        "label_sequence_length": "Sequence Length",
        "label_dropout_rate": "Dropout Rate",
        "label_hidden_neuron": "Hidden Neuron",
        "label_number_of_layers": "Number of Layers",
        "label_batch_size": "Batch Size",
        "label_gru_units": "GRU Units",
        "label_num_heads": "Num Heads",
        "label_model_dimension": "Model Dimension",
        "label_num_layers": "Num Layers",

        "table_hydro_model": "Hydro Model",
        "table_rating_curve": "Rating Curve",
        "table_basic_model": "Basic Model",
        "table_predict_rainfall_model": "Predict Rainfall Model",
        "table_scenario_model": "Scenario Model",

        "graph_water_level": "Water Level",
        "graph_flow_rate": "Flow Rate",
        "graph_observed_waterlevel": "Observed Water Level",
        "graph_predicted_waterlevel": "Predicted Water Level",
        "graph_waterlevel_prediction_nse": "Water Level Prediction (NSE={})",
        "graph_flood_watch": "Flood Watch",
        "graph_flood_warning": "Flood Warning",
        "graph_current_time": "Current Time :",

        "common_please_select": "Please select",

        "title_model_path_set": "Model Path Settings",
        "title_model_add": "Add Model",
        "title_model_del": "Delete Model",

        "label_model_name": "Model Name :",
        #"label_python_path": "Python Path :",
        "label_model_format_file_path": "Model Format File Path :",

        "btn_select_python": "Select Path",
        "btn_select_folder": "Select Folder",
        "btn_ok": "OK",
        "btn_cancel": "Cancel",

        "title_warning": "Warning",
        "msg_enter_model_name": "Please enter the model name.",
        "msg_select_python_path": "Please select the Python path.",
        "msg_invalid_python_path": "The selected Python path is invalid.",
        "msg_select_model_format_file_path": "Please select the model format file path.",
        "msg_invalid_model_format_file_path": "The selected model format file path is invalid.",

        "msg_closeEvent": "Are you sure you want to exit?",
        "msg_env_enter_model_name": "Please enter a model name.",
        "msg_env_already_model_name": "A model with the same name already exists.",
        "msg_env_delete_model": "Delete the model {}?",

        "combo_item_normalized_data": "Normalized Data",
        "combo_item_missing_value_correction": "Missing Value Correction",
        "combo_item_outlier_filtered_data": "Outlier-Filtered Data",

        "placeholder_enter_station_id": "Enter Station ID",
        "placeholder_search_field": "Search Field",
        
        "msg_data_count_differs_by_station": "The number of data points varies by observation station.",
        "msg_failed_to_get_data": "Failed to get {data}.",
        "msg_target_leadtime_count_mismatch": "The number of data points for the Target station does not match the number of lead times.",
        "msg_target_file_exists": "[Target Point File Information]\n{path} exists.",
        "msg_target_file_not_exists": "[Target Point File Information]\n{path} does not exist.",
        "msg_reference_file_exists": "[Reference Point File Information]\n{path} exists.",
        "msg_reference_file_not_exists": "[Reference Point File Information]\n{path} does not exist.",
        "msg_nse_file_exists": "[NSE File Information]\n{path} exists.",
        "msg_nse_file_not_exists": "[NSE File Information]\n{path} does not exist.",
        "msg_validation_file_exists": "[Validation File Information]\n{path} exists.",
        "msg_validation_file_not_exists": "[Validation File Information]\n{path} does not exist.",
        "msg_inference_file_exists": "[Inference File Information]\n{path} exists.",
        "msg_inference_file_not_exists": "[Inference File Information]\n{path} does not exist.",
        "msg_dataset_file_exists": "[{year} Dataset File Information]\n{path} exists.",
        "msg_dataset_file_not_exists": "[{year} Dataset File Information]\n{path} does not exist.",
        "msg_failed_to_process": "Failed to process {obs_id}. {error}",
        "msg_processing_error": "An error occurred during processing.",
        "msg_database_save_error": "Database save error for {obs_id}: {error}",
        "msg_layer_not_found": "Cannot find the layer '{layer_name}'.",
        "msg_layer_loaded": "'{layer_name}' layer was loaded successfully.",
        "msg_no_available_tablespace": "[Database Settings] No available tablespace.",
        "filedlg_select_python_executable": "Select Python Executable",
        "filedlg_select_model_format_folder": "Select Model Format Folder",
        "msg_change_service_start_type": "[Note] Change the start type of {service_name} from 'Disabled' to 'Manual'.",
        "msg_service_start_failed": "[Error] {service_name} service failed to start.",
        "msg_failed_to_load_tablespaces": "[Database Settings] Failed to load tablespaces: {error}",
        "filedlg_open_file": "Open File",
        "filedlg_save_file": "Save File",
        "filedlg_save_html_file": "Save HTML File",
        "msg_layer_connection_already_disconnected": "[Query Database] The layer connection has already been disconnected.",
        "msg_openapi_layer_connection_already_disconnected": "[Query OpenAPI] Layer connection is already disconnected.",
        "msg_confirm_import_file": "Do you want to import the selected file into the {table_name} table?",
        "msg_query_database_error": "[Query Database] Error occurred: {error}",
        "msg_query_database_exception": "[Query Database] {error}",
        "error_load_python_console": "[Load Model] Failed to load the Python console: {error}",
        "error_init_database_date": "[Database Settings] Failed to initialize the database date range: {error}",
        "error_load_shapefiles": "[Layer Settings] Failed to load shapefiles: {error}",
        "error_terminate_db_import_thread": "[Database Import] Failed to terminate the import thread: {error}",
        "error_wait_dataset_threadpool": "[Dataset] Failed to wait for the thread pool to finish: {error}",

        "summary_target_point": "Target Point",
        "summary_reference_point": "Reference Point",
        "summary_period": "Period",
        "summary_lead_time": "Lead Time",
        "summary_preprocessing_options": "Preprocessing Options",

        "error_empty_target_point": "Target point information is empty.",
        "error_empty_lead_time": "Lead time information is empty.",
        "error_invalid_query_period": "The query period information is invalid.",
        "error_no_target_waterlevel_data": "There is no target water level data in the selected search period.",
        "error_insufficient_target_waterlevel_data": "There is not enough target water level data for the selected lead time.\nPlease shorten the end date or reduce the maximum lead time.",
        "error_dataset_precheck_database": "A database error occurred during the dataset pre-check.\n{error}",
        "error_dataset_precheck": "An error occurred during the dataset pre-check.\n{error}",
        #"error_empty_lead_time": "[Dataset Creation] Lead time information is empty.",
        "error_remove_existing_file": "Failed to remove the existing file:\n{path}\n\n{error}",

        "error_model_data_file_not_found": "[Model Creation] Data file not found: {path}",
        "error_model_data_load_failed": "[Model Creation] Failed to load data: {error}",
        "error_prediction_file_not_found": "[Prediction and Performance Analysis] File not found: {model_name}",
        "error_prediction_file_read_failed": "[Prediction and Performance Analysis] An error occurred while reading the file: {error}",

        "log_load_model": "[Prediction and Performance Analysis] Loading model...",
        "log_model_loaded_successfully": "[Prediction and Performance Analysis] Model loaded successfully.",
        "log_loaded_model_type": "[Prediction and Performance Analysis] Loaded model type: {model_type}",
        "log_model_type_not_detected": "[Prediction and Performance Analysis] The model type could not be detected.",
        "log_model_load_failed": "[Prediction and Performance Analysis] Failed to load the model.",

        "log_prediction_start": "[Prediction and Performance Analysis] Starting prediction...",
        "log_prediction_failed": "[Prediction and Performance Analysis] Prediction failed.",
        "error_prediction_file_or_python_not_found": "[Prediction and Performance Analysis] The file path is incorrect, or the Python executable could not be found.",
        "error_prediction_exception": "[Prediction and Performance Analysis] An exception occurred: {error}",
        "log_prediction_output": "=== Prediction Log ===",
        "error_prediction_log_read_failed": "[Prediction and Performance Analysis] Failed to read the prediction log: {error}",
        "error_prediction_no_test_data": "[Prediction and Performance Analysis] Prediction could not be performed because there is no test data.",
        "log_prediction_completed": "[Prediction and Performance Analysis] Prediction has been completed.",
        "error_prediction_failed_exit_code": "[Prediction and Performance Analysis] Prediction failed. Exit code: {exit_code}",

        "log_realtime_model_summary": "[Real-time Prediction] Model Summary",
        "log_realtime_error_occurred": "[Real-time Prediction] Error occurred: {error}",
        "log_realtime_python_not_found": "[Real-time Prediction] The file path is incorrect, or the Python executable could not be found.",
        "log_realtime_exception_occurred": "[Real-time Prediction] Exception occurred: {error}",

        "log_realtime_mariadb_error": "[Real-time Prediction] MariaDB error occurred: {error}",
        "log_realtime_exception_occurred": "[Real-time Prediction] Exception occurred: {error}",

        "log_realtime_start": "[Real-time Prediction] Start real-time prediction.",
        "log_realtime_model_summary": "[Real-time Prediction] Model Summary",
        "log_realtime_finish": "[Real-time Prediction] Real-time prediction has finished.",
        "log_realtime_warning": "[Real-time Prediction] Warning occurred: {error}",
        "log_realtime_error_occurred": "[Real-time Prediction] Error occurred: {error}",
        "log_realtime_python_not_found": "[Real-time Prediction] The file path is incorrect, or the Python executable could not be found.",
        "log_realtime_exception_occurred": "[Real-time Prediction] Exception occurred: {error}",

        "label_dataset_graph_period_info": "Dataset Graph Period Information ({0} ~ {1})",
        "log_dashboard_model_parameter_not_found": "[Dashboard] Unable to retrieve the parameter information for the model at {target_point}.",

        "log_combo_item_not_found": "[Notice] '{text}' does not exist in the combo box.",
        "log_all_files_processed_successfully": "All files processed successfully.",

        "log_predict_rain_column_count_mismatch": "Column count mismatch in predict_rain for {obs_id}. Expected: {expected}, Actual: {actual}",
        "log_temp_table_creation_error": "Error creating temporary table for {obs_id}: {error}",
        "log_load_data_error": "LOAD DATA error for {obs_id}: {error}",
        "log_database_merge_error": "Database merge error for {obs_id}: {error}",
        "log_unsupported_column_count": "Unsupported column count for {obs_id}: {column_count}",

        "log_initialize_file_read_failed": "[File Initialization] Failed to read file: {error}",
        "log_initialize_file_min_columns": "[File Initialization] CSV must contain at least 2 columns.",
        "error_fewer_than_two_columns": "Fewer than 2 columns detected.",
        "log_threehr_interpolation_error": "[Interpolation] An error occurred: {error}",

        #################################################

        "msg_bayesian_optimization_failed_exit_code": "Bayesian Optimization failed.\nExit code: {}",
        "msg_optimal_parameters_not_applied_files_deleted": "The optimal parameters were not applied to the model creation controls.\n"
            "The optimization result files have been deleted.",
        "msg_optimal_result_creation_controls": "Do you want to apply these optimal parameters to the model creation controls?",
        "msg_optimal_result_complete": "Bayesian Optimization has been completed.\n\n",
        "msg_optimization_files_already_exist_run_again": "Model optimization result files already exist.\n\n"
            "Do you want to delete the existing results and run the optimization again?",
        "msg_failed_delete_existing_optimization_files": "Failed to delete the existing model optimization result files.\n\n{}",

        'msg_info': 'Information',
        'msg_tr_self_current_language_message_key': 'tr(self.current_language, message_key)',
        'msg_warning': 'Warning',
        'msg_error': 'Error',
        'title_app': 'KICT FLOOD AI',
        'title_db_connection_error': 'Database (MariaDB) Connection Error',
        'msg_administrator_privileges_required_start_mariadb_service_continue_without': 'Administrator privileges are '
                                                                                        'required to start the MariaDB '
                                                                                        'service.\n'
                                                                                        '\n'
                                                                                        'Continue without a database?',
        'msg_run_program_without_mariadb_some_features_may_limited': 'Run the program without MariaDB.\n'
                                                                     'Some features may be limited.',
        'msg_qgis_will_closed_restart_administrator_mode': 'QGIS will be closed. Please restart it in administrator '
                                                           'mode.',
        'title_db_service_start_error': 'Database (MariaDB) Service Start Error',
        'msg_f_exception_occurred_while_starting_mariadb_service_e': 'An exception occurred while starting the MariaDB '
                                                                     'service :\n'
                                                                     '{}',
        'msg_check_database_connection_information_again': 'Please check the database connection information again.',
        'title_performance_check': 'Performance Check',
        'msg_failed_load_file_format_e': 'Failed to load file.\n{}',
        'msg_selected_file_empty': 'The selected file is empty.',
        'msg_not_find_datetime_column': 'Could not find a datetime column.',
        'msg_selected_file_format_different_only_prediction_result_csv': 'The selected file format is different.\n'
                                                                         '\n'
                                                                         'Only prediction result CSV files can be '
                                                                         'loaded.',
        'msg_no_valid_datetime_rows_found': 'No valid datetime rows were found.',
        'msg_no_valid_datetime_rows_found_after_filtering': 'No valid datetime rows were found after filtering.',
        'msg_not_find_observed_water_level_column': 'Could not find an observed water level column.',
        'title_extreme_flood_scenario_creation': 'Extreme Flood Scenario Creation',
        'msg_there_no_information_selected_folder': 'There is no information for the selected folder.',
        'msg_enter_parent_folder_path_modelparam': 'Please enter the parent folder path for ModelParam..',
        'msg_enter_folder_path_extreme_flood_scenario_model': 'Please enter the folder path for the Extreme Flood '
                                                              'Scenario model.',
        'msg_enter_save_path_scenario_dataset': 'Please enter the save path for the scenario dataset.',
        'msg_enter_folder_containing_observation_data_files': 'Please enter the folder containing the observation data '
                                                              'files.',
        'msg_select_extreme_flood_scenario_name': 'Please select a Extreme Flood Scenario name.',
        'msg_cannot_extract_water_level_observation_station_code_scenario': 'Cannot extract the water level '
                                                                            'observation station code from the '
                                                                            'scenario name.',
        'msg_f_cannot_load_scenario_module_make_scenario_import': 'Cannot load the scenario module (.make_scenario).\n'
                                                                  '{}',
        'msg_scenario_creation_complete_want_check_file': 'Scenario Creation is complete. Do you want to check the '
                                                          'file?',
        'msg_msg': 'msg',
        'title_query_database': 'Query Database',
        'msg_select_type_table_query': 'Please select the type of table to query.',
        'msg_war_msg': 'war_msg',
        'msg_no_layer_selected': 'No layer selected.',
        'title_query_openapi': 'Query OpenAPI',
        'msg_select_type_data_query': 'Please select the type of data to query.',
        'title_dashboard': 'Dashboard',
        'msg_set_folder_path_data_display_on_dashboard': 'Please set the folder path for the data to display on the '
                                                         'dashboard.',
        'msg_no_folder_selected': 'No folder selected.',
        'title_data_file_db_import': 'Data File DB Import',
        'msg_there_no_data_import': 'There is no data to import.',
        'msg_msgimport': 'msgImport',
        'msg_operation_been_canceled': 'The operation has been canceled.',
        'msg_if_duplicate_data_exists_want_update': 'If duplicate data exists, do you want to update it?',
        'msg_e': 'e',
        'msg_all_files_been_saved': 'All files have been saved.',
        'msg_invalid_internet_connection': 'Invalid internet connection.',
        'msg_data_result_errdata': "data['result'][errData]",
        'msg_data_errdata': 'data[errData]',
        'msg_errmsg': 'errMsg',
        'msg_enter_observation_station_number_query': 'Please enter the observation station number to query.',
        'msg_start_date_cannot_later_than_end_date': 'The start date cannot be later than the end date.',
        'msg_query_period_limited_one_day': 'Query period is limited to one day.',
        'msg_query_period_limited_one_month': 'Query period is limited to one month.',
        'msg_there_no_data_save': 'There is no data to save.',
        'msg_data_been_saved_successfully': 'The data has been saved successfully',
        'msg_select_table_query': 'Please select a table to query.',
        'msg_save_table_query': 'Please select a table to save.',
        'msg_enter_observation_station_id_query': 'Please enter the observation station id to query.',
        'msg_no_data_found': 'No data found.',
        'title_dataset_creation': 'Dataset Creation',
        'msg_select_only_one_target_point': 'Select only one Target Point.',
        'msg_no_feature_selected': 'No feature selected',
        'msg_enter_select_target_point_information_query': 'Please enter or select the Target Point information to '
                                                           'query.',
        'msg_enter_select_reference_point_information_query': 'Please enter or select the Reference Point information '
                                                              'to query.',
        'msg_target_point_also_included_reference_point_list_remove': 'The Target Point is also included in the '
                                                                      'Reference Point list.\n'
                                                                      'Please remove the duplicated point and try '
                                                                      'again.',
        'msg_set_period_query': 'Please set the period for the query.',
        'msg_no_rainfall_point_information_selected_so_dry_weather': 'No rainfall point information is selected, so '
                                                                     'the dry weather event data cannot be checked.',
        'msg_recheck_information_preprocessing_option_delete_dry_weather_event': 'Please recheck the information for '
                                                                                 'the preprocessing option: Delete Dry '
                                                                                 'Weather Event.',
        'msg_recheck_input_information_preprocessing_option_delete_dry_weather': 'Please recheck the input information '
                                                                                 'for the preprocessing option: Delete '
                                                                                 'Dry Weather Event.\n'
                                                                                 '(Example - enter a positive integer '
                                                                                 'value)',
        'msg_enter_both_water_level_change_time_threshold_no': 'Please enter both Water Level Change time and '
                                                               'threshold for the No-Rainfall Event option.',
        'msg_enter_water_level_change_time_as_positive_integer': 'Please enter the Water Level Change time as a '
                                                                 'positive integer.',
        'msg_enter_water_level_change_threshold_as_number_greater': 'Please enter the Water Level Change threshold as '
                                                                    'a number greater than or equal to 0.',
        'msg_no_rainfall_point_information_selected_so_cumulative_rainfall': 'No rainfall point information is '
                                                                             'selected, so cumulative rainfall data '
                                                                             'cannot be generated.',
        'msg_recheck_information_preprocessing_option_generate_cumulative_rainfall': 'Please recheck the information '
                                                                                     'for the preprocessing option: '
                                                                                     'Generate Cumulative Rainfall.',
        'msg_recheck_input_information_preprocessing_option_generate_cumulative_rainfall': 'Please recheck the input '
                                                                                           'information for the '
                                                                                           'preprocessing option: '
                                                                                           'Generate Cumulative '
                                                                                           'Rainfall.\n'
                                                                                           '(Example - enter 3,6 for '
                                                                                           '3-hour and 6-hour '
                                                                                           'periods)\n'
                                                                                           'Only positive integers are '
                                                                                           'allowed, and '
                                                                                           'duplicate/empty values are '
                                                                                           'not allowed.',
        'msg_recheck_information_preprocessing_option_period_filtering': 'Please recheck the information for the '
                                                                         'preprocessing option: Period Filtering.',
        'msg_no_watershed_information_selected_so_predictive_rainfall_data': 'No watershed information is selected, so '
                                                                             'predictive rainfall data cannot be '
                                                                             'generated.',
        'msg_recheck_information_preprocessing_option_generate_predictive_rainfall': 'Please recheck the information '
                                                                                     'for the preprocessing option: '
                                                                                     'Generate Predictive Rainfall.',
        'msg_recheck_input_information_preprocessing_option_generate_predictive_rainfall': 'Please recheck the input '
                                                                                           'information for the '
                                                                                           'preprocessing option: '
                                                                                           'Generate Predictive '
                                                                                           'Rainfall.\n'
                                                                                           '(Example - enter 3,6 for '
                                                                                           '3-hour and 6-hour '
                                                                                           'periods)\n'
                                                                                           'Only positive integers are '
                                                                                           'allowed, and '
                                                                                           'duplicate/empty values are '
                                                                                           'not allowed.',
        'msg_set_lead_time_information_query': 'Please set the lead time information to query.',
        'msg_recheck_time_series_lead_time_settings': 'Please recheck the Time Series lead time settings.',
        'msg_lead_time_interval_cannot_greater_than_target_lead': 'The lead time interval cannot be greater than the '
                                                                  'target lead time.',
        'msg_f_e': '{}',
        'msg_dataset_creation_been_completed': 'Dataset Creation has been completed.',
        'msg_precheck_msg': 'precheck_msg',
        'msg_f_create_dataset_configured_information_summary': 'Create dataset with the configured information??\n{}',
        'msg_save_without_null_values_yes_save_excluding_null': 'Save without NULL values?\n'
                                                                '(Yes-Save excluding NULL values, No-Save all data)',
        'msg_dataset_file_been_saved_want_check_file': 'Dataset file has been saved. Do you want to check the file?',
        'msg_no_data_query': 'No data to query',
        'title_model_creation': 'Model Creation',
        'msg_f_check_contents_modelparameter_file_following_keys_missing': 'Please check the contents of the '
                                                                           'ModelParameter file.\n'
                                                                           "The following keys are missing '{}'.",
        'msg_f_want_update_current_value_current_value_origin': 'Do you want to update the current value?\n'
                                                                '\n'
                                                                'Current Value : {}\n'
                                                                'New value : {}',
        'msg_select_dataset_file': 'Please select a dataset file.',
        'msg_dataset_file_contains_null_values': 'The dataset file contains NULL values.',
        'msg_select_folder_save_model': 'Please select a folder to save the model.',
        #'msg_enter_model_name': 'Please enter the model name.',
        'msg_select_model': 'Please select a model.',
        'msg_python_executable_path_selected_model_empty': 'The Python executable path for the selected model is '
                                                           'empty.',
        'msg_model_folder_path_selected_model_empty': 'The model folder path for the selected model is empty.',
        'msg_check_hidden_layer_input_values_example_32_64': 'Please check the Hidden Layer input values.\n'
                                                             '(Example: 32,64,32)',
        'msg_enter_values_according_number_hidden_layers': 'Please enter values according to the number of Hidden '
                                                           'Layers.',
        'msg_enter_values_matching_number_hidden_layers': 'Please enter values matching the number of Hidden Layers.',
        'msg_select_data_columns_use_model_creation': 'Please select the data columns to use for model creation.',
        'msg_select_lead_time_column_use_model_creation': 'Please select the lead time column to use for model '
                                                          'creation.',
        'msg_enter_standard_watershed_code_information': 'Please enter the standard watershed code information.',
        'msg_standard_watershed_code_ws_information_missing_not_selected': 'The standard watershed code (WS) '
                                                                           'information is missing or not selected in '
                                                                           'the dataset.',
        'msg_standard_watershed_code_predict_ws_information_applying_observed': 'The standard watershed code '
                                                                                '(Predict_WS) information for applying '
                                                                                'observed rainfall is missing or not '
                                                                                'selected in the dataset.',
        'msg_use_apply_realtime_predict_ws_set_observed_data': 'To use Apply Realtime Predict_WS, please set the '
                                                               'observed data folder path in System Configuration → '
                                                               'Dashboard Settings.',
        'msg_f_no_test_data_set_want_proceed_anyway': 'No Test data is set. Do you want to proceed anyway?',
        'msg_reset_test_data': 'Please reset the Test data.',
        'title_model_optimization': 'Model Optimization',
        'msg_f_bayesian_py_file_not_found_bayesian_py': 'bayesian.py file not found.\n\n{}',
        'msg_python_executable_path_invalid': 'The Python executable path is invalid.',
        'msg_model_folder_path_invalid': 'The model folder path is invalid.',
        'msg_f_bayesian_py_not_found_script_path': 'bayesian.py was not found.\n\n{}',
        'msg_f_failed_start_bayesian_optimization_e': 'Failed to start Bayesian Optimization.\n\n{}',
        'msg_f_bayesian_optimization_failed_exit_code_exit_code': 'Bayesian Optimization failed.\n'
                                                                  'Exit code: {}\n'
                                                                  '\n'
                                                                  'Log path:\n'
                                                                  '{}\n'
                                                                  '\n'
                                                                  'Last log:\n'
                                                                  '{}',
        'msg_f_best_params_csv_not_found_best_params': 'best_params.csv was not found.\n\n{}',
        'msg_f_optimization_trials_csv_not_found_trials_path': 'optimization_trials.csv was not found.\n\n{}',
        'msg_f_best_params_csv_empty_best_params_path': 'best_params.csv is empty.\n\n{}',
        'msg_summary': 'summary',
        'msg_optimal_parameters_been_applied_model_creation_controls': 'The optimal parameters have been applied to '
                                                                       'the model creation controls.',
        'msg_f_optimal_parameters_saved_only_best_params_csv': 'The optimal parameters were saved only.\n'
                                                               '\n'
                                                               'best_params.csv:\n'
                                                               '{}\n'
                                                               '\n'
                                                               'optimization_trials.csv:\n'
                                                               '{}',
        'msg_f_failed_read_apply_best_params_csv_e': 'Failed to read or apply best_params.csv.\n\n{}',
        'msg_folder_already_exists_want_overwrite': 'The folder already exists. Do you want to overwrite it?',
        'msg_path_contains_korean_characters_change_path_use_only': 'The path contains Korean characters. Please '
                                                                    'change the path to use only English characters.',
        'msg_f_watershed_rainfall_column_not_found_dataset_missing': 'The watershed rainfall column was not found in '
                                                                     'the dataset.\n'
                                                                     '\n'
                                                                     'Missing column : {}',
        'msg_f_makemodel_py_not_found_script_path': 'makemodel.py was not found.\n\n{}',
        'msg_f_modelparam_txt_not_found_modelpath': 'modelParam.txt was not found.\n\n{}',
        'msg_model_created_successfully_however_there_no_test_data': 'The model was created successfully.\n'
                                                                     '\n'
                                                                     'However, there was no test data, so '
                                                                     'prediction/evaluation results were not '
                                                                     'generated.\n'
                                                                     '\n'
                                                                     'Do you want to view the results folder?',
        'msg_model_been_successfully_created_want_view_results': 'The model has been successfully created. Do you want '
                                                                 'to view the results?',
        'msg_f_model_creation_failed_exit_code_exit_code': 'Model creation failed. Exit code: {}',
        'msg_dataset_file_does_not_contain_target_point_information': 'The dataset file does not contain Target point '
                                                                      'information.',
        'msg_dataset_file_does_not_contain_leadtime_information': 'The dataset file does not contain Leadtime '
                                                                  'information.',
        'msg_dataset_file_does_not_contain_reference_point_information': 'The dataset file does not contain Reference '
                                                                         'point information.',
        'title_prediction_performance_analysis': 'Prediction and Performance Analysis',
        'msg_select_model_folder_path': 'Please select the model folder path.',
        'msg_select_folder_path_save_prediction_results': 'Please select the folder path to save the prediction '
                                                          'results.',
        'msg_enter_name_prediction_results': 'Please enter the name for the prediction results.',
        'msg_select_data_use_prediction': 'Please select the data to use for prediction.',
        'msg_number_data_items_used_model_data_prediction_not': 'The number of data items used in the model and the '
                                                                'data for prediction do not match.',
        'msg_types_data_used_model_data_selected_prediction_not': 'The types of data used in the model and the data '
                                                                  'selected for prediction do not match.',
        'msg_select_lead_time_use_prediction': 'Please select the lead time to use for prediction.',
        'msg_number_lead_times_used_model_lead_times_selected': 'The number of lead times used in the model and the '
                                                                'lead times selected for prediction do not match.',
        'msg_types_lead_times_used_model_lead_times_selected': 'The types of lead times used in the model and the lead '
                                                               'times selected for prediction do not match.',
        'msg_standard_watershed_codes_used_model_missing_not_selected': 'The standard watershed codes used in the '
                                                                        'model are missing or not selected in the '
                                                                        'dataset.',
        'msg_f_file_modelname_not_found': 'The file {} could not be found.',
        'msg_f_result_stderr': '{}',
        'msg_f_e_stderr': '{}',
        'msg_f_file_path_incorrect_python_executable_not_found': 'The file path is incorrect, or the Python executable '
                                                                 'could not be found.',
        'msg_f_failed_prepare_result_folder_e': 'Failed to prepare the result folder.\n\n{}',
        'msg_prediction_not_performed_because_there_no_test_data': 'Prediction could not be performed because there is '
                                                                   'no test data in the selected dataset.',
        'msg_prediction_been_completed_want_view_results': 'Prediction has been completed. Do you want to view the '
                                                           'results?',
        'msg_f_prediction_failed_exit_code_exit_code': 'Prediction failed. Exit code: {}',
        'title_realtime_prediction': 'Real-time Prediction',
        'msg_select_prediction_model_use_real_time_prediction': 'Please select the prediction model to use for '
                                                                'real-time prediction.',
        'msg_select_folder_path_save_real_time_prediction_results': 'Please select the folder path to save the '
                                                                    'real-time prediction results.',
        'msg_number_data_items_varies_by_observation_station': 'The number of data items varies by observation '
                                                               'station.',
        'msg_sure_want_exit': 'Are you sure you want to exit?',
        'title_dataset_graph_save': 'Dataset Graph Save',
        'msg_there_no_graph_html_file_save': 'There is no graph HTML file to save.',
        'msg_f_html_saved_file_path': 'HTML saved to:\n{}',
        'msg_f_failed_save_html_file_e': 'Failed to save HTML file.\n{}',
        'msg_there_no_graph_save': 'There is no graph to save.',
        'title_dashboard_file_information': 'Dashboard File Information',
        'title_dashboard_report': 'Dashboard Report',
        'msg_no_format_file_found_report_creation': 'No format file found for report creation.',
        'msg_failed_open_hwp_check_whether_hancom_office_hwp': 'Failed to open HWP. Please check whether Hancom Office '
                                                               'HWP is installed or verify the file.',
        'title_create_database': 'Create Database',
        'info_create_database': 'Enter the database name.',
        'msg_enter_name': 'Please enter a name.',
        'msg_check_predictrain_table': 'Create Prediction Rainfall Table (predict_rain)',
        'msg_input_can_contain_only_letters_numbers_underscores': 'Input can contain only letters, numbers, or '
                                                                  'underscores(_).',
        'msg_f_name_database_already_exists': "'{}' database already exists.",
        'msg_f_error_occurred_e': 'Error occurred.\n{}',
        #'title_bayesian_optimization': 'Bayesian Optimization',
        'msg_f_label_text_min_value_cannot_greater_than': '{} Min value cannot be greater than Max value.',
        'title_connect_database': 'Database Connection Settings',
        'info_connect_database': 'Configure the Database connection information.',
        'btn_fileselectdlg_apply': 'Apply',
        'btn_fileselectdlg_cancel': 'Cancel',
        'btn_fileselectdlg_selFile': 'Select File',

        "msg_input_error_title": "Input Error",
        "msg_fileselectdlg_select_layer_file": "Please select a layer file.",
        "msg_fileselectdlg_layer_file_not_exist": "The selected layer file does not exist.",
        "msg_fileselectdlg_invalid_shapefile": "Please select a valid shapefile (*.shp).",
        "msg_fileselectdlg_select_station_code": "Please select a station code field.",
        "msg_fileselectdlg_field_info_not_loaded": "Layer field information could not be loaded.",
        "msg_fileselectdlg_failed_load_layer": "Failed to load the selected layer.",
        "msg_fileselectdlg_station_field_not_exist": "The selected station code field does not exist in the layer.",
        
        #################################################

        # TabWidget
        "tab_dataset": "Dataset Creation",
        "tab_model": "Model Creation",
        "tab_predict": "Prediction and Performance Analysis",
        "tab_realtime": "Real-time Prediction",
        "tab_performanceCheck": "Performance Check",
        "tab_dataMng": "Data Management",
        "tab_makrScenario": "Extreme Flood Scenario Creation",
        "tab_envSet": "System Configuration",
        
        "tab_ds_original": "Original Dataset",
        "tab_ds_preprocessing": "Preprocessed Dataset",
        
        "tab_3": "Import Data File to Database",
        "tab_5": "Query Database",
        "tab_7": "Query OpenAPI",
        
        "tab_6": "Hydrological Model (Storage Function Method)",
        
        # QToolbox
        "tb_set_point": "Step 1: Select Observation Site",
        "tb_set_period": "Step 2: Select Search Period and Lead Time",
        "tb_set_option": "Option: Data Preprocessing",

        # QPushButton
        "btn_dataset_activelayer_targetpoint": "Select",
        "btn_dataset_activelayer_waterlevel": "Select",
        "btn_dataset_activelayer_rainfall": "Select",
        "btn_dataset_activelayer_discharge": "Select",
        "btn_dataset_activelayer_daminlet": "Select",
        "btn_dataset_activelayer_damrelease": "Select",
        "btn_dataset_activelayer_tidelevel": "Select",
        "btn_dataset_activelayer_watershed": "Select",
        "btn_dataset_selLayer_init": "Reset",
        
        "btn_makeDataset": "Make Dataset",
        "btn_ds_origin_prev": "Prev Page",
        "btn_ds_origin_next": "Next Page",
        "btn_ds_preset_prev": "Prev Page",
        "btn_ds_preset_next": "Next Page",
        "btn_showDatasetGraph": "View Graph",
        "btn_saveDataset": "Save Dataset as CSV",
        
        "btn_param_datasetPath": "Select",
        "btn_param_savePath": "Select",
        "btn_dl_dataset_prev": "Prev Page",
        "btn_dl_dataset_next": "Next Page",
        "btn_makemodel_optimize": "Model Optimization",
        "btn_dl_loadParameter": "Load Model Parameter File",
        "btn_makeModel": "Create Model",
        
        "btn_selectDataFile": "Select",
        "btn_selectModelFile": "Select",
        "btn_load_dl_dataset_prev": "Prev Page",
        "btn_load_dl_dataset_next": "Next Page",
        "btn_dl_reloadModel": "Run Prediction",
        
        "btn_selectRealtimeModelFile": "Select Model",
        "btn_saveRealtimeModelPath": "Select",
        "btn_realtimeModel": "Start Real-Time Prediction",
        
        "btn_selectPerformResultFile": "Select",
        "btn_perform_createWindow": "Create Window",
        
        "btn_data_loadDataFile": "Select",
        "btn_dataSave": "Save",
        "btn_datasearchGispoint": "Select from GIS",
        "btn_dataSearch": "Search",
        "btn_apisearchGispoint": "Select from GIS",
        "btn_apiSearch": "Search",
        "btn_apiSaveCSV": "Save as CSV",
        
        "btn_sceModel_select_scenario_model_folder": "Select Folder",
        "btn_sceModel_select_modelParam_folder": "Select Folder",
        "btn_sceModel_select_dataset_folder": "Select Folder",
        "btn_sceModel_select_observ_folderpath": "Select Folder",
        "btn_makeScenario_SFM": "Create Scenario Dataset",
        
        "btn_envSet_database": "Database Settings",
        "btn_envSet_layerinfo": "Layer Settings",
        "btn_envSet_dashboard": "Dashboard Settings",
        "btn_envSet_python": "Model Path Settings",

        "btn_connect_mariadb": "Edit Database Connection Information",
        "btn_create_tablespace": "Create New Database",
        
        "btn_env_lyr_waterlevel": "Change Layer",
        "btn_env_lyr_rainfall": "Change Layer",
        "btn_env_lyr_discharge": "Change Layer",
        "btn_env_lyr_daminlet": "Change Layer",
        "btn_env_lyr_damrelease": "Change Layer",
        "btn_env_lyr_tidelevel": "Change Layer",
        "btn_env_lyr_watershed": "Change Layer",
        "btn_env_lyr_rivernetwork": "Change Layer",
        "btn_env_lyr_default": "Set as Default Layer",
        
        "btn_dash_modify_model": "Edit",
        "btn_dash_modify_year": "Edit",
        "btn_dash_modify_riverRegion": "Edit",
        "btn_dash_modify_modelName": "Edit",
        
        "btn_config_model_add": "Add",
        "btn_config_model_del": "Delete",
        "btn_check_pythonPath": "Check Python Path",

        # QGroupBox        
        "gb_datasetLayer": "Dataset Creation Info",
        "groupBox_8": "Search Period Settings",
        "groupBox_9": "Leadtime Settings",
        "groupBox_13": "Exclude Dry-Day Events",
        "groupBox_20": "Generate Cumulative Rainfall",
        "groupBox_10": "Generate Forecast Rainfall",
        
        "groupBox_makemodel_dataset": "Please select the dataset file.",
        "groupBox_makemodel_param": "Please enter the model parameter information.",
        "groupBox_makemodel_save": "Please set the save path information.",
        
        "groupBox_predict_dataset": "Please select the dataset file.",
        "groupBox_predict_model": "Please select a model.",
        "groupBox_predict_dl": "Please select dataset columns and lead times.",
        "groupBox_predict_save": "Please set the save path information.",
        
        "groupBox_31": "Please configure real-time prediction execution settings.",
        
        "groupBox": "Please select the result file.",
        
        "groupBox_18": "Database Settings",
        "groupBox_30": "Set Database Connection Information.",
        "groupBox_25": "Set Database Information.",
        "groupBox_19": "Layer Settings",
        "groupBox_22": "Dashboard Settings",
        "groupBox_26": "Set the Parent Folder Containing the Model.",
        "groupBox_27": "Add, Edit, or Delete Items for Each Level.",
        "groupBox_28": "Set the Folder Containing Observation Data.",
        "groupBox_29": "Python Path Settings",   

        # QChceckBox
        "ckb_ds_preProcessing_noRain": "Set Time Before and After Excluding Dry-Day Events",
        "ckb_ds_preProcessing_checkWL": "Check Waterlevel",
        "sp_ds_preProcessing_noRainVal": "Generate Cumulative Rainfall",
        "ckb_ds_preProcessing_month": "Refilter Period Within Query Range",
        "ckb_ds_preProcessing_predictRainVal": "Generate Forecast Rainfall",
        
        "ckb_data_option2": "Missing Value Correction",
        "ckb_data_option1": "Outlier Removal",
        "label_8": "IP Time(H)",
        "ckb_data_saveFile": "File",
        "ckb_data_saveDbImport": "Database",

        # QRadioButton
        "rb_dataset_periodDate": "Query by Detailed Period",
        "rb_dataset_periodYear": "Query by Year",
        "rb_ds_leadtimeBasic": "By Leadtiime",
        "rb_ds_leadtimeTimeseries": "By Timeseries",
        "rb_ds_preProcessing_month": "Specific Month",
        "rb_ds_preProcessing_date": "Specific Period",
        
        "rb_perform_manual": "Manual",
        "rb_perform_automatic": "Automatic",
        "perform_option_peak_waterlevel": "Based on Peak Water Level",
        "perform_option_peak_rainfall": "Based on Peak Rainfall",
        
        "rb_observ_folderpath_db": "From Database",
        "rb_observ_folderpath_file": "From File",

        # QLabel
        "lbl_env_waterlevel_2": "Select Database :",
        "label_93": "Folder Path :",
        "label_97": "[Note] Information for Level 4 (Target Point) is retrieved using the folder information within the region selected at Level 3.",
        "label_99": "Folder Path :",

        "label_14": """
        <html>
        <head>
        <style type="text/css">
        p {
            margin-top: 0px;
            margin-bottom: 4px;
            margin-left: 0px;
            margin-right: 0px;
            white-space: pre-wrap;
        }
        </style>
        </head>
        <body style="font-family:'Gulim'; font-size:9pt; font-weight:400; font-style:normal;">
        <p>※ Click the <span style="font-weight:600;">Check Python Path</span> button, and in the popup window, select the <span style="font-family:'Courier New';">checkpythonpath.py</span> file, then run it using <span style="font-weight:600;">Python (associated program)</span>.</p>
        <p>※ Please check the Python executable path displayed in the Command Prompt (cmd) window.</p>
        <p>※ If Python is not installed, the associated program option will not appear.</p>
        <p>※ To perform <span style="font-weight:600;">River Water Level Prediction Model Development</span>, make sure to download and install Python.</p>
        </body>
        </html>
        """,
        
        "label_84": "Select Query Year :",
        "label_83": "Target Time :",
        "label_88": "Time Interval :",
        
        "label_92": "Set Real-Time Prediction Start Time :",
        
        "label_85": "Start Time :",
        "label_86": "Period :",
        "label_102": "Range :",
        
        "label_78": "Select Data File :",
        "label_82": "Settings :",
        "label_81": "Select Table",
        "label_80": "Data Quality Control (Options)",
        "label_100": "Output Options",
        
        "label0_2": "Data Table :",
        "label01_3": "Station ID :",
        "label01_4": "Query Period :",
        
        "label_60": """
            <html><head/><body><p>Fetch data from the Han River Flood Control Office (water level, flow rate, dam inflow/outflow – 10-minute intervals, up to 1 month) and the Korea Hydrographic and Oceanographic Agency</p><p>(tide level – 1-minute intervals, up to 1 day) via OpenAPI.</p></body></html>
            """,      
        "label_2": "Data Type :",
        "label_3": "Station ID :",    
        "label_4": "Query Period :",  
        "label_23": "Save the query results as a CSV file. The observation station code will be used as the default file name.",
        
        "label_130": "Scenario Model Folder Path :",
        "label_133": "Scenario Name :",
        "label_128": "Parent Folder Path of ModelParam :",
        "label_129": "Basin Name :",
        "label_131": "Dataset Save Path :",
        "label_134": "Observed Data Application Method :",

    } # type: ignore
}


def tr(lang, key):
    return TEXT.get(lang, TEXT["en"]).get(key, key)