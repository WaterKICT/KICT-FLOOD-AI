import os
import ast
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from random import *
from pickle import dump
from matplotlib import gridspec        
import re
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from io import BytesIO

# 대형홍수시나리오
class MakeScenario():

    def __init__(self, options):

        self.options = options

        '''options = {
            'modelParam_path' : modelParam_path,
            'River_area': River_area,
            'scenario_model_folder_path': scenario_model_folder_path,
            'dataset_path': dataset_path,
            'scenario_name': scenario_name,
            'folder_name': folder_name,
            'bfromfile' : self.rb_observ_folderpath_file.isChecked(),
            'bfromdb' : self.rb_observ_folderpath_db.isChecked(),
            'obs_folder' : self.txt_observ_folderpath.text().strip(),
            'db_config' : self.db_config,
            'db_tablespace' : self.db_tablespace,
        }'''


    # [대형홍수시나리오] 수문학적모형(저류함수법) 생성
    def Storage_Function_Method(self):
        
        # (x)Model 폴더패스 
        #model_folder_path = self.txt_model_forder_path.text()
        # (o)ModelParam 상위폴더패스 ---> 모델파라메터가있는폴더 패스
        modelParam_path = self.options['modelParam_path']
        # (o)ModelParam 하위패스 (권역명-Han)
        River_area = self.options['River_area']
        # (o)대형홍수시나리오모델 패스 ----> 시나리오모델이 있는패스 (out, in, scenario_file)
        scenario_model_folder_path = self.options['scenario_model_folder_path']
        # (o)데이터셋 패스(x) ---> 저장경로패스 
        dataset_path = self.options['dataset_path']
        # (x)모델(년도)
        #model_version = self.txt_model_version.text()
        # (o)대형홍수시나리오명 (obscode 수위관측소명) ---> 대형홍수시나리오모델패스 안에 존재하는 폴더
        scenario_name = self.options['scenario_name']

        bfromfile = self.options['bfromfile']
        bfromdb = self.options['bfromdb']

        obs_folder = self.options['obs_folder']
        db_config = self.options['db_config']
        db_tablespace = self.options['db_tablespace']
        folder_name = self.options['folder_name']

        dataset_origin_path = ''

        # alpha 값 
        alpha = os.listdir(os.path.join(scenario_model_folder_path, scenario_name))
        alpha = [word.lower() for word in alpha]

        for k in range(len(alpha)):
            if ('alpha=' not in alpha[k]) :
                alpha[k] = None

        alpha = [item for item in alpha if item != None ]
        alpha = [f"alpha={float(item.split('=')[1]):.3f}" for item in alpha]

        alpha = [float(word.strip('alpha=')) for word in alpha]
        alpha.sort(reverse=False)

        
        X_data_training_column_list_index = 9999
        seq_length_index = 9999
        RF_cum_index = 9999
        WS_cum_index = 9999
        Y_data_training_column_list_index = 9999
        data_path_index = 9999

        print('*******************************************Making dataset is starting  : {}'.format(folder_name))

        # ModelParam 상위폴더패스 >> ModelParam 하위패스 (권역명-Han) 폴더 안에있는 
        # modelParam_관측소코드.txt 파일에서 정보 가져오기 
        modelParam_file_path = os.path.join(modelParam_path, River_area , 'modelParam_{}.txt'.format(folder_name))

        if not os.path.exists(modelParam_file_path):            
            return False, f"File not found : {modelParam_file_path}"

        # read modelParam file
        modelParam_list = None
        #with open(modelParam_file_path, "r", encoding="utf-8-sig") as f:
        with open(modelParam_file_path, "r", encoding="cp949") as f:
            modelParam_list = f.readlines()

            # search index included in X_data_training_column_list
            for i in range(len(modelParam_list)):
                if 'X_data_training_column_list' in modelParam_list[i] :
                    X_data_training_column_list_index = i

            # search index included in seq_length
            for i in range(len(modelParam_list)):
                if 'seq_length' in modelParam_list[i] :
                    seq_length_index = i

            # search index included in RF_cum
            for i in range(len(modelParam_list)):
                if 'RF_cum' in modelParam_list[i] :
                    RF_cum_index = i
                    
            # search index included in Predict_WS
            for i in range(len(modelParam_list)):
                if 'Predict_WS' in modelParam_list[i] :
                    WS_cum_index = i

            # search index included in X_data_training_column_list
            for i in range(len(modelParam_list)):
                if 'Y_data_training_column_list' in modelParam_list[i] :
                    Y_data_training_column_list_index = i
            
            # search index included in data_path
            for i in range(len(modelParam_list)):                
                if 'data_path' in modelParam_list[i] :
                    data_path_index = i            

        # data_path
        from pathlib import Path
        data_path_line = modelParam_list[data_path_index].strip('\n')
        dataset_origin_path_str = data_path_line[data_path_line.index(',') + 1:].strip()
        dataset_origin_path = Path(r"{}".format(dataset_origin_path_str))
        print("data_path =", dataset_origin_path)

        X_data_training_column_list = modelParam_list[X_data_training_column_list_index].strip('\n')[modelParam_list[X_data_training_column_list_index].index(',')+1:]
        post_list = eval(X_data_training_column_list)
        print('X_data_training_coumn = {}'.format(post_list)) 

        Y_data_training_column_list = modelParam_list[Y_data_training_column_list_index].strip('\n')[modelParam_list[Y_data_training_column_list_index].index(',')+1:]
        lead_cols = eval(Y_data_training_column_list)
        print('Y_data_training_column = {}'.format(lead_cols))

        seq_length = int(modelParam_list[seq_length_index].strip('\n')[modelParam_list[seq_length_index].index(',')+1:])
        print('seq_length = {}'.format(str(seq_length)))

        # RF_cum
        cumulative_hour = 3
        cumulative_hours = []
        if RF_cum_index != 9999 :
            cumulative_hour = int(modelParam_list[RF_cum_index].strip('\n')[modelParam_list[RF_cum_index].index('RF_cum')+6:modelParam_list[RF_cum_index].index('RF_cum')+7])
            lists = modelParam_list[RF_cum_index].strip('\n')          
            re_lists = []  
            match = re.search(r"\[.*\]", lists)
            if match:
                array_str = match.group(0)  
                re_lists = ast.literal_eval(array_str)  
            cumulative_hours = sorted(set(
                int(re.search(r'RF_cum(\d+)_', col).group(1))
                for col in re_lists
                if col.startswith('RF_cum')
            ))
        else :
            cumulative_hour = 3    
            cumulative_hours.append(3)

        # Predict_WS
        predict_hour = 3
        predict_hours = []
        if WS_cum_index != 9999 :

            predict_hour = int(modelParam_list[WS_cum_index].strip('\n')[modelParam_list[WS_cum_index].index('Predict_WS')+10:modelParam_list[WS_cum_index].index('Predict_WS')+11])
            
            lists = modelParam_list[WS_cum_index].strip('\n')          
            re_lists = []  
            match = re.search(r"\[.*\]", lists)
            if match:
                array_str = match.group(0)  
                re_lists = ast.literal_eval(array_str)  
            predict_hours = sorted(set(
                int(re.search(r'Predict_WS(\d+)_', col).group(1))
                for col in re_lists
                if col.startswith('Predict_WS')
            ))
        else :
            predict_hour = 0    

       
        target_point = []
        WL_point = []
        Discharge = []
        RF_point = []
        Inflow = []
        Release = []
        Watershed_RF = []
        Tide = []

        for i in range(len(post_list)):
            obs_code = post_list[i].strip().split('_')[-1]
            if 'Target' in post_list[i] :
                target_point.append(obs_code)
            if 'WL' in post_list[i] :
                WL_point.append(obs_code)
            if 'DC' in post_list[i] :
                Discharge.append(obs_code)
            if 'RF' in post_list[i] :
                RF_point.append(obs_code)
            if 'DI' in post_list[i] :
                Inflow.append(obs_code)
            if 'DR' in post_list[i] :
                Release.append(obs_code)
            if 'WS' in post_list[i] :
                Watershed_RF.append(obs_code)
            if 'TE' in post_list[i] :
                parts = post_list[i].strip().split('_')  
                result = '_'.join(parts[1:])
                Tide.append(result)

            target_point = sorted(set(target_point), key=lambda x: target_point.index(x))
            WL_point = sorted(set(WL_point), key=lambda x: WL_point.index(x))
            RF_point = sorted(set(RF_point), key=lambda x: RF_point.index(x))
            Discharge = sorted(set(Discharge), key=lambda x: Discharge.index(x))
            Inflow = sorted(set(Inflow), key=lambda x: Inflow.index(x))
            Release = sorted(set(Release), key=lambda x: Release.index(x))
            Watershed_RF = sorted(set(Watershed_RF), key=lambda x: Watershed_RF.index(x))
            Tide = sorted(set(Tide), key=lambda x: Tide.index(x))

        print('*******************Making the parameter is completed*****************')

        print('Target point= ', target_point)
        print('WL_point= ',WL_point)
        print('RF_point= ', RF_point)
        print('Discharge= ', Discharge)
        print('Release= ', Release)
        print('Inlet= ', Inflow)
        print('Watershed_RF= ', Watershed_RF)
        print('Tide= ', Tide)        
        print('cumulative_hour = {}'.format(str(cumulative_hour)), cumulative_hours)
        print('predict_hour = {}'.format(str(predict_hour)), predict_hours)

        # Find Waterlevel_filename from Tatget and WL point
        
        # observation data_path
        pluginPath = os.path.dirname(__file__)
        modelObservation_folder = pluginPath + "/station_info"  
        observ_path = os.path.join(modelObservation_folder, 'WL.csv')
        if not os.path.exists(observ_path):            
            return False, f"[Extreme Flood Scenario Creation] File not found: {observ_path}"

        waterlevel_post_list = pd.read_csv(observ_path, encoding="cp949", dtype={'StationCo': str})

        wl_post_name_list = []
        dc_post_name_list = []
        wl_code_name_list = []

        for i in range(len(target_point)) :
            for j in range(len(waterlevel_post_list)) :
                if target_point[i] == waterlevel_post_list['StationCo'][j] : 
                    wl_post_name_list.append(waterlevel_post_list['Name_Kor'][j])
                    wl_code_name_list.append(target_point[i] )
                    
        for i in range(len(WL_point)) :
            for j in range(len(waterlevel_post_list)) :
                if WL_point[i] == waterlevel_post_list['StationCo'][j] : 
                    wl_post_name_list.append(waterlevel_post_list['Name_Kor'][j])
                    wl_code_name_list.append(WL_point[i] )

        for i in range(len(Discharge)) :
            for j in range(len(waterlevel_post_list)) :
                if Discharge[i] == waterlevel_post_list['StationCo'][j] : 
                    dc_post_name_list.append(waterlevel_post_list['Name_Kor'][j])

        ############################################################

        # make file list
        output_folder_path = os.path.join(scenario_model_folder_path,scenario_name,'Correction','Output')
        if not os.path.exists(output_folder_path):
            return False, f"[Extreme Flood Scenario Creation] File not found: {output_folder_path}"
        
        outputfile_list = os.listdir(output_folder_path)
        wl_post_file_list =[]
        dc_post_file_list =[]

        wl_post_file_list = [
            next((f for f in outputfile_list if f'c_{post}' in f), None)
            for post in wl_post_name_list
        ]        

        dc_post_file_list = [
            next((f for f in outputfile_list if f'c_{post}' in f), None)
            for post in dc_post_name_list
        ]        

        print ('wl_post_file_list - ', wl_post_file_list)
        print ('dc_post_file_list = ', dc_post_file_list)

        # making the rainfall data
        precipitation_path = os.path.join(scenario_model_folder_path,scenario_name,'Correction','input','precipitation.hyd')
        if not os.path.exists(precipitation_path):
            return False, f"[Extreme Flood Scenario Creation] File not found: {precipitation_path}"

        precipitation = pd.read_fwf(precipitation_path)
        precipitation_column_list = list(precipitation.columns)
        precipitation_column_list[0] = "Date"
        precipitation.columns = precipitation_column_list

        # 날짜컬럼만 따로 추출해놓기
        date_column = pd.DataFrame(precipitation['Date'])
        
        # Unless save_path folder exist, make the save_path folder (시나리오폴더 생성)
        save_path = os.path.join(dataset_path,River_area,folder_name,'scenario')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # RF_point=  ['10014120', '10014260', '13024010']
        # precipitation.hyd (강수량)에서 가져온 데이터 (precipitation) 에서,
        # 강우관측소지점(RF_point)이 존재하는 컬럼만 추출 (관측소가 강수파일에 없다면 빈값으로 세팅)        
        #measure_precipitation = precipitation[['Date']+RF_point] <- 문제점 : 강우지점이 없을때 오류발생
        valid_RF = [col for col in RF_point if col in precipitation.columns]
        missing_RF = [col for col in RF_point if col not in precipitation.columns]

        measure_precipitation = precipitation[['Date'] + valid_RF].copy()  

        for col in missing_RF:
            measure_precipitation[col] = np.nan        

        measure_precipitation = measure_precipitation[['Date'] + RF_point]      

        RF_header = ['Date'] + ['RF_{}'.format(rf) for rf in RF_point]
        measure_precipitation.columns = RF_header   

        rf_columns = RF_header[1:]
        measure_precipitation[rf_columns] = measure_precipitation[rf_columns].fillna(0.0)
        measure_precipitation[rf_columns] = measure_precipitation[rf_columns].clip(lower=0.0)


        # making the cumulative rainfall data (데이터가없을때는 0처리함)
        
        print("Make cumulative rainfall with multiple alphas and cumulative_hours")

        # alpha 리스트에 1.0 포함
        alpha_rain = [1.0] + alpha

        RF_cum_total = pd.DataFrame()

        for a in alpha_rain:
            print(f"Processing alpha = {a}")
            
            tmp_precipitation = measure_precipitation.iloc[:, 1:] * a
            RF_cum_df = pd.DataFrame(date_column.copy())
            for rp in RF_point:
                RF_cum_df[f'RF_{rp}'] = tmp_precipitation[f'RF_{rp}']
            
            for ch in cumulative_hours:
                for rp in RF_point:
                    temp_RF = tmp_precipitation[f'RF_{rp}'].fillna(0)
                    temp_RF_cum = [temp_RF[max(0, i-(ch*6-1)):i+1].sum() for i in range(len(temp_RF))]
                    RF_cum_df[f'RF_cum{ch}_{rp}'] = temp_RF_cum
            
            RF_cum_df['Date'] = RF_cum_df['Date'].str.replace('@',' ')
            
            if a == 1.0:
                file_name = os.path.join(save_path, 'alpha_1.0_precipitation.csv')
            else:
                file_name = os.path.join(save_path, f'alpha_{a}_precipitation.csv')

            if not RF_cum_df.drop(columns=["Date"]).dropna(how="all").empty:
                RF_cum_df.to_csv(file_name, index=False)
            
            if a > 1.0:
                RF_cum_total = pd.concat([RF_cum_total, RF_cum_df], ignore_index=True)

        # 전체 합 저장
        if not RF_cum_total.drop(columns=["Date"]).dropna(how="all").empty:
            RF_cum_total.to_csv(os.path.join(save_path, 'total_precipitation.csv'), index=False)

        print("Cumulative rainfall processing complete")            
                
        # calculate the start time of flood event
        start_time = measure_precipitation['Date'][0]
        end_time = measure_precipitation['Date'].iloc[-1]

        print ('start_time - ', start_time)
        print ('end_time - ', end_time)
        
        # 초기 데이터셋 생성
        WS_Dataset = pd.DataFrame({'Date': pd.to_datetime(measure_precipitation['Date'].str.replace('@',' '))})
        REL_Dataset = pd.DataFrame({'Date': WS_Dataset['Date']})
        IN_Dataset = pd.DataFrame({'Date': WS_Dataset['Date']})
        TIDE_Dataset = pd.DataFrame({'Date': WS_Dataset['Date']})

        # target  type 1 : 표준화.     2 : 결측보정.     3 : 이상치보정  
        WS_Type = 3     # 2023년 이후 자료는 선택가능
        REL_Type = 2    # 2023년 이후 자료는 선택가능
        IN_Type = 2     # 2023년 이후 자료는 선택가능
        TIDE_Type = 2

        # case1. 파일에서 데이터 가져오기                        
        if bfromfile:

            year = int(start_time[:4])

            # set_obs_data_path
            obs_data_path = os.path.join(obs_folder, f'data_{year}')
            if not os.path.exists(obs_data_path):
                obs_data_path = os.path.join(obs_folder, 'data_2011-2022')

            # make_obs_data
            def read_and_format(filepath, prefix, col_type, obs_name):
                if not os.path.exists(filepath):
                    print(f"[Extreme Flood Scenario Creation] File not found : {filepath}")
                    return pd.DataFrame(index=pd.to_datetime(WS_Dataset['Date']), columns=[f"{prefix}_{obs_name}"])
                
                try:
                    df = pd.read_csv(filepath, header=None, usecols=[0, col_type])
                    df.columns = ['Date', f"{prefix}_{obs_name}"]
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df.set_index('Date')
                except Exception as e:
                    print(f"[Extreme Flood Scenario Creation] Failed to read file : {filepath}, {e}")
                    return pd.DataFrame(index=pd.to_datetime(WS_Dataset['Date']), columns=[f"{prefix}_{obs_name}"])

            # watershed
            try:
                ws_frames = []
                for obs in Watershed_RF:
                    file = os.path.join(obs_data_path, 'watershed', obs + '.csv')
                    ws_frames.append(read_and_format(file, "WS", WS_Type, obs))
                date_index = pd.to_datetime(WS_Dataset["Date"])
                base = WS_Dataset.set_index("Date")
                aligned_frames = [df.reindex(date_index) for df in ws_frames]
                WS_Dataset = pd.concat([base] + aligned_frames, axis=1).reset_index()
                WS_Dataset.fillna(0, inplace=True)  

            except Exception as e:
                print(f"[Extreme Flood Scenario Creation] Failed to process Watershed data : {e}")

            # damrelease
            try:
                rel_frames = []
                for obs in Release:
                    file = os.path.join(obs_data_path, 'damrelease', obs + '.csv')
                    rel_frames.append(read_and_format(file, "DR", REL_Type, obs))                            
                date_index = pd.to_datetime(REL_Dataset["Date"])
                base = REL_Dataset.set_index("Date")
                aligned_frames = [df.reindex(date_index) for df in rel_frames]
                REL_Dataset = pd.concat([base] + aligned_frames, axis=1).reset_index()
                REL_Dataset.fillna(0, inplace=True)

            except Exception as e:
                print(f"[Extreme Flood Scenario Creation] Failed to process Dam Release data : {e}")

            # daminlet
            try:
                inflow_frames = []
                for obs in Release:
                    file = os.path.join(obs_data_path, 'daminlet', obs + '.csv')
                    inflow_frames.append(read_and_format(file, "DI", IN_Type, obs))
                date_index = pd.to_datetime(IN_Dataset["Date"])
                base = IN_Dataset.set_index("Date")
                aligned_frames = [df.reindex(date_index) for df in inflow_frames]
                IN_Dataset = pd.concat([base] + aligned_frames, axis=1).reset_index()
                IN_Dataset.fillna(0, inplace=True)

            except Exception as e:
                print(f"[Extreme Flood Scenario Creation] Failed to process Dam Inflow data : {e}")

            # tidelevel
            try:
                tide_frames = []
                for obs in Tide:
                    file = os.path.join(obs_data_path, 'tidelevel', obs + '.csv')
                    tide_frames.append(read_and_format(file, "TE", TIDE_Type, obs))                
                date_index = pd.to_datetime(TIDE_Dataset["Date"])
                base = TIDE_Dataset.set_index("Date")
                aligned_frames = [df.reindex(date_index) for df in tide_frames]
                TIDE_Dataset = pd.concat([base] + aligned_frames, axis=1).reset_index()
                TIDE_Dataset.fillna(0, inplace=True)
                
            except Exception as e:
                print(f"[Extreme Flood Scenario Creation] Failed to process Tidelevel data : {e}")

        
        # case2. 데이터베이스에서 가져오기       
        if bfromdb:

            import mariadb

            # DB 접속
            try:
                db_config["database"] = db_tablespace
                conn = mariadb.connect(**db_config)
                cur = conn.cursor()
            except mariadb.Error as e:
                print(f"[Extreme Flood Scenario Creation] Failed to connect to the database : {e}")
                conn = None

            # make_obs_data
            def get_data_obs(obs_list, dataset, table_name, col_prefix, data_type, start_time=None, end_time=None):
                if conn is None:
                    return dataset
                
                for obs_id in obs_list:
                    try:
                        if start_time is None or end_time is None:
                            tmp_df = pd.DataFrame({'Date': dataset['Date'], f'{col_prefix}_{obs_id}': [0]*len(dataset)})
                        else:
                            start = start_time.replace("@", " ")
                            end = end_time.replace("@", " ")
                            sql_date = f"DT_DATE >= '{start}:00' AND DT_DATE <= '{end}:00'"
                            query = f"SELECT DT_DATE as Date, DT_DATA, MI_DATA, OI_DATA FROM {table_name} WHERE OBS_ID='{obs_id}' AND ({sql_date}) ORDER BY DT_DATE"
                            cur.execute(query)
                            result = cur.fetchall()
                            
                            if result:
                                tmp_df = pd.DataFrame(result, columns=['Date','DT_DATA','MI_DATA','OI_DATA']).iloc[:, [0,data_type]]
                                tmp_df.columns = ['Date', f'{col_prefix}_{obs_id}']
                                tmp_df['Date'] = pd.to_datetime(tmp_df['Date'])
                            else:
                                tmp_df = pd.DataFrame({'Date': dataset['Date'], f'{col_prefix}_{obs_id}': [0]*len(dataset)})

                        dataset = pd.merge(dataset, tmp_df, on='Date', how='left')
                    except Exception as e:
                        print(f"[Extreme Flood Scenario Creation] {table_name} {obs_id} Error : {e}")
                return dataset

            WS_Dataset = get_data_obs(Watershed_RF, WS_Dataset, 'watershed', 'WS', WS_Type, start_time, end_time)
            REL_Dataset = get_data_obs(Release, REL_Dataset, 'damrelease', 'DR', REL_Type, start_time, end_time)
            IN_Dataset = get_data_obs(Inflow, IN_Dataset, 'daminlet', 'DI', IN_Type, start_time, end_time)
            TIDE_Dataset = get_data_obs(Tide, TIDE_Dataset, 'tidelevel', 'TE', TIDE_Type, start_time, end_time)

            # make_predict_ws
            if len(predict_hours) > 0 and conn:
                for pred_hour in predict_hours:
                    pred_startDt = pd.to_datetime(start_time.replace("@", " ")) + timedelta(hours=pred_hour)
                    pred_endDt = pd.to_datetime(end_time.replace("@", " ")) + timedelta(hours=pred_hour)
                    sql_predictDate = f"date_format(DT_DATE, '%Y-%m-%d %H:%i') between '{pred_startDt}' and '{pred_endDt}'"
                    
                    for obs_id in Watershed_RF:
                        try:
                            query = f"SELECT DT_DATE as Date, DT_DATA, MI_DATA, OI_DATA FROM watershed WHERE OBS_ID='{obs_id}' AND ({sql_predictDate}) ORDER BY DT_DATE"
                            cur.execute(query)
                            result = cur.fetchall()
                            
                            if result:
                                tmp_df = pd.DataFrame(result, columns=['Date','DT_DATA','MI_DATA','OI_DATA']).iloc[:, [0,WS_Type]]
                                tmp_df.columns = ['Date', f'Predict_WS{pred_hour}_{obs_id}']
                                tmp_df['Date'] = pd.to_datetime(tmp_df['Date'])
                                tmp_df['Date'] = tmp_df['Date'] - timedelta(hours=6)
                            else:
                                tmp_df = pd.DataFrame({'Date': WS_Dataset['Date'], f'Predict_WS{pred_hour}_{obs_id}': [0]*len(WS_Dataset)})
                            
                            WS_Dataset = pd.merge(WS_Dataset, tmp_df, on='Date', how='left')
                            
                            # 첫행 bfill 처리
                            col_name = f'Predict_WS{pred_hour}_{obs_id}'
                            WS_Dataset.loc[0, col_name] = WS_Dataset[col_name].bfill().iloc[0]
                            
                        except Exception as e:
                            print(f"[Extreme Flood Scenario Creation] predict {obs_id} {pred_hour} Time error : {e}")
            
            if conn:
                conn.close()

        # save_1.0
        if not WS_Dataset.drop(columns=["Date"]).dropna(how="all").empty:            
            WS_Dataset.to_csv(os.path.join(save_path,'alpha_1.0_watershed.csv'), index=False)
        if not REL_Dataset.drop(columns=["Date"]).dropna(how="all").empty:            
            REL_Dataset.to_csv(os.path.join(save_path,'alpha_1.0_release.csv'), index=False)
        if not IN_Dataset.drop(columns=["Date"]).dropna(how="all").empty:            
            IN_Dataset.to_csv(os.path.join(save_path,'alpha_1.0_inflow.csv'), index=False)
        if not TIDE_Dataset.drop(columns=["Date"]).dropna(how="all").empty:            
            TIDE_Dataset.to_csv(os.path.join(save_path,'alpha_1.0_tidelevel.csv'), index=False)


        # making the watershed data for each alpha
        WS_Dataset_total = pd.DataFrame()
        DI_Dataset_total = pd.DataFrame()
        REL_Dataset_total = pd.DataFrame()        
        TIDE_Dataset_total = pd.DataFrame()

        for j in range(len(alpha)) :

            # calculate the watershed data for each alpha
            if not WS_Dataset.empty:
                tmp_WS_Dataset = WS_Dataset[WS_Dataset.columns[1:]]*alpha[j]
                WS_Dataset_alpha = pd.concat([WS_Dataset['Date'],tmp_WS_Dataset],axis=1)                
                
                if not WS_Dataset_alpha.drop(columns=["Date"]).dropna(how="all").empty:      
                    WS_Dataset_alpha.to_csv(os.path.join(save_path,'alpha_{}_watershed.csv'.format(alpha[j])),index=False)
                WS_Dataset_total = pd.concat([WS_Dataset_total,WS_Dataset_alpha], ignore_index=True) 
            
            # calculate the release data for each alpha
            if not REL_Dataset.empty:
                tmp_REL_Dataset = REL_Dataset[REL_Dataset.columns[1:]]
                REL_Dataset_alpha = pd.concat([REL_Dataset['Date'],tmp_REL_Dataset],axis=1)

                if not REL_Dataset_alpha.drop(columns=["Date"]).dropna(how="all").empty:   
                    REL_Dataset_alpha.to_csv(os.path.join(save_path,'alpha_{}_release.csv'.format(alpha[j])),index=False)
                REL_Dataset_total = pd.concat([REL_Dataset_total,REL_Dataset_alpha], ignore_index=True) 
            
            # calculate the inflow data for each alpha
            if not IN_Dataset.empty:
                tmp_DI_Dataset = IN_Dataset[IN_Dataset.columns[1:]]
                DI_Dataset_alpha = pd.concat([IN_Dataset['Date'],tmp_DI_Dataset],axis=1)

                if not DI_Dataset_alpha.drop(columns=["Date"]).dropna(how="all").empty: 
                    DI_Dataset_alpha.to_csv(os.path.join(save_path,'alpha_{}_inflow.csv'.format(alpha[j])),index=False)
                DI_Dataset_total = pd.concat([DI_Dataset_total,DI_Dataset_alpha], ignore_index=True) 
                
            # calculate the tidelevel data for each alpha
            if not TIDE_Dataset.empty:
                tmp_TIDE_Dataset = TIDE_Dataset[TIDE_Dataset.columns[1:]]
                TIDE_Dataset_alpha = pd.concat([TIDE_Dataset['Date'],tmp_TIDE_Dataset],axis=1)

                if not TIDE_Dataset_alpha.drop(columns=["Date"]).dropna(how="all").empty: 
                    TIDE_Dataset_alpha.to_csv(os.path.join(save_path,'alpha_{}_tidelevel.csv'.format(alpha[j])),index=False)
                TIDE_Dataset_total = pd.concat([TIDE_Dataset_total,TIDE_Dataset_alpha], ignore_index=True) 
            
        # save the total file
        if not WS_Dataset_total.drop(columns=["Date"]).dropna(how="all").empty: 
            WS_Dataset_total.to_csv(os.path.join(save_path,'total_watershed.csv'.format(alpha[j])),index=False)
        if not REL_Dataset_total.drop(columns=["Date"]).dropna(how="all").empty: 
            REL_Dataset_total.to_csv(os.path.join(save_path,'total_release.csv'.format(alpha[j])),index=False)
        if not DI_Dataset_total.drop(columns=["Date"]).dropna(how="all").empty: 
            DI_Dataset_total.to_csv(os.path.join(save_path,'total_inflow.csv'.format(alpha[j])),index=False)
        if not TIDE_Dataset_total.drop(columns=["Date"]).dropna(how="all").empty: 
            TIDE_Dataset_total.to_csv(os.path.join(save_path,'total_tidelevel.csv'.format(alpha[j])),index=False)


        # making the discharge data

        discharge_data = pd.DataFrame()
        if len(Discharge) > 0:
            discharge_data = pd.DataFrame()

            for i, file_name in enumerate(dc_post_file_list):

                discharge_path = os.path.join(scenario_model_folder_path, scenario_name, "Correction", "Output", file_name)

                if not os.path.exists(discharge_path):
                    print(f"[Extreme Flood Scenario Creation] File not found : {discharge_path}")
                    continue

                # read file
                with open(discharge_path, "r", encoding="cp949") as f:
                    tmp_lines = f.readlines()

                # find the index of data start
                skip_index = None
                for k, line in enumerate(tmp_lines):
                    if "------------------------------------" in line:
                        skip_index = k
                        break

                if skip_index is None:
                    print(f"[Extreme Flood Scenario Creation] Separator (---) not found in the file : {file_name}")
                    continue

                # read waterlevel from output file
                data_split = [x.strip().split() for x in tmp_lines[skip_index+1:]]
                tmp_discharge = pd.DataFrame(data_split)

                # set column
                if len(tmp_discharge.columns) == 9:
                    tmp_discharge.columns = [
                        "Date", "Avg_rainfall", "Cum_rainfall", "Forecast_inflow",
                        f"DC_{Discharge[i]}", "Measure_discharge",
                        "Forecasted_stage", "Measure_stage", "Remarks"
                    ]
                elif len(tmp_discharge.columns) == 8:
                    tmp_discharge.columns = [
                        "Date", "Avg_rainfall", "Cum_rainfall", "Forecast_inflow",
                        f"DC_{Discharge[i]}", "Measure_discharge",
                        "Forecasted_stage", "Measure_stage"
                    ]
                else:
                    print(f"[Extreme Flood Scenario Creation] The number of columns in the file ({len(tmp_discharge.columns)}) is different : {file_name}")
                    continue

                # merge
                if i == 0:
                    discharge_data = pd.concat(
                        [discharge_data, tmp_discharge[["Date", f"DC_{Discharge[i]}"]]],
                        axis=1
                    )
                else:
                    discharge_data = pd.concat(
                        [discharge_data, tmp_discharge[[f"DC_{Discharge[i]}"]]],
                        axis=1
                    )

            # set date column type (replace @, u -> ' ')
            if not discharge_data.empty and "Date" in discharge_data.columns:
                discharge_data["Date"] = discharge_data["Date"].str.replace(r"[@u]", " ", regex=True)
                discharge_data["Date"] = pd.to_datetime(discharge_data["Date"], errors="coerce")

            # save data
            if not discharge_data.empty:
                discharge_data.to_csv(os.path.join(save_path, "alpha_1.0_discharge.csv"), index=False)

        # making the discharge data for each alpha

        discharge_total = pd.DataFrame()
        
        if len(Discharge) > 0:

            for j, a in enumerate(alpha):
                discharge_alpha = pd.DataFrame()
                print(f"---------------- Alpha = {a:.3f}")

                temp_list = []  

                for i, file_name in enumerate(dc_post_file_list):
                    file_path = os.path.join(scenario_model_folder_path, scenario_name, f"Alpha={a:.3f}", "Output", file_name)

                    if not os.path.exists(file_path):
                        print(f"[Extreme Flood Scenario Creation] File not found : {file_path}")
                        continue

                    # read file
                    with open(file_path, "r", encoding="cp949") as f:
                        lines = f.readlines()

                    # find the index of data start
                    skip_index = None
                    for k, line in enumerate(lines):
                        if "------------------------------------" in line:
                            skip_index = k
                            break

                    if skip_index is None:
                        print(f"[Extreme Flood Scenario Creation] Data start line(---) not found in the file : {file_name}")
                        continue

                    # read discharge from output file
                    data_split = [x.strip().split() for x in lines[skip_index+1:]]
                    tmp_discharge = pd.DataFrame(data_split)

                    # set column
                    if len(tmp_discharge.columns) == 9:
                        tmp_discharge.columns = [
                            "Date", "Avg_rainfall", "Cum_rainfall", "Forecast_inflow",
                            f"DC_{Discharge[i]}", "Measure_discharge",
                            "Forecasted_stage", "Measure_stage", "Remarks"
                        ]
                    elif len(tmp_discharge.columns) == 8:
                        tmp_discharge.columns = [
                            "Date", "Avg_rainfall", "Cum_rainfall", "Forecast_inflow",
                            f"DC_{Discharge[i]}", "Measure_discharge",
                            "Forecasted_stage", "Measure_stage"
                        ]
                    else:
                        print(f"[Extreme Flood Scenario Creation] The number of columns in the file ({len(tmp_discharge.columns)}) is different : {file_name}")
                        continue

                    if i == 0:
                        temp_list.append(tmp_discharge[["Date", f"DC_{Discharge[i]}"]])
                    else:
                        temp_list.append(tmp_discharge[[f"DC_{Discharge[i]}"]])

                # merge
                if temp_list:
                    discharge_alpha = pd.concat(temp_list, axis=1)

                    # set date column type (replace @, u -> ' ')
                    discharge_alpha["Date"] = discharge_alpha["Date"].str.replace(r"[@u]", " ", regex=True)
                    discharge_alpha["Date"] = pd.to_datetime(discharge_alpha["Date"], errors="coerce")

                    # 개별 저장
                    discharge_alpha.to_csv(
                        os.path.join(save_path, f"alpha_{a:.3f}_discharge.csv"),
                        index=False,
                        encoding="utf-8-sig"
                    )

                    # 전체 합치기
                    discharge_total = pd.concat([discharge_total, discharge_alpha], ignore_index=True)

            # 전체 저장
            if not discharge_total.empty:
                discharge_total.to_csv(os.path.join(save_path, "total_discharge.csv"), index=False, encoding="cp949")

        ############################################################

        # making the target waterlevel data    

        waterlevel_total = pd.DataFrame()
        target_leadtime_total = pd.DataFrame()
        no_WL_list = []

        # set_leadtime

        if lead_cols and "min" in lead_cols[0]:   # ex) Leadtime_10min, Leadtime_20min ...
            Leadtime = [int(col.replace("Leadtime_", "").replace("min", "")) for col in lead_cols]
            leadtime_header = [f"Leadtime_{x}min" for x in Leadtime]

        elif lead_cols and "H" in lead_cols[0]:   # ex) Leadtime_1H, Leadtime_2H ...
            Leadtime = [int(col.replace("Leadtime_", "").replace("H", "")) for col in lead_cols]
            leadtime_header = [f"Leadtime_{x}H" for x in Leadtime]

        else: #없을경우min으로 간주.
            Leadtime = [int(col.replace("Leadtime_", "")) for col in lead_cols]
            leadtime_header = [f"Leadtime_{x}min" for x in Leadtime]

        # alpha 1.0

        print('---------------- Alpha = 1.0 ----------------')
        waterlevel = pd.DataFrame()
        alpha_folder = os.path.join(scenario_model_folder_path, scenario_name, 'Correction', 'Output')

        for i, wl_file in enumerate(wl_post_file_list):
            file_path = os.path.join(alpha_folder, wl_file)

            if not os.path.exists(file_path):
                print(f"[Extreme Flood Scenario Creation] File not found : {file_path}")
                continue
            
            with open(file_path, 'r', encoding='cp949') as f:
                tmp_lines = f.readlines()

            skip_index = next((k for k, line in enumerate(tmp_lines) if '------------------------------------' in line), None)
            if skip_index is None:
                print(f"[Extreme Flood Scenario Creation] {wl_file} does not contain data separator.")
                continue

            tmp_lines = tmp_lines[skip_index + 1:]
            data_split = [x.strip().split() for x in tmp_lines]
            tmp_df = pd.DataFrame(data_split)

            if len(tmp_df.columns) == 9:
                if i == 0:
                    tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                    'Measure_discharge', f'Target_{target_point[i]}', 'Measure_stage', 'Remarks']
                else:
                    tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                    'Measure_discharge', f'WL_{WL_point[i-1]}', 'Measure_stage', 'Remarks']
            elif len(tmp_df.columns) == 8:
                if i == 0:
                    tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                    'Measure_discharge', f'Target_{target_point[i]}', 'Measure_stage']
                else:
                    tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                    'Measure_discharge', f'WL_{WL_point[i-1]}', 'Measure_stage']
            else:
                print(f"[Extreme Flood Scenario Creation] {wl_file} has unexpected number of columns: {len(tmp_df.columns)}")
                continue

            if i == 0:
                waterlevel = pd.concat([waterlevel, tmp_df[['Date', f'Target_{target_point[i]}']]], axis=1)
                target_data = waterlevel.iloc[:, 1].astype(float).reset_index(drop=True)

                # Leadtime 생성
                leadtime_data_list = []
                for lt in Leadtime:
                    if "min" in lead_cols[0]:
                        shift = int(lt / 10)  # 10분 단위 데이터
                    elif "H" in lead_cols[0]:
                        shift = int((lt * 60) / 10)  # 시간 단위 → 10분 간격 데이터 기준
                    else:#없을경우 min으로 간주
                        shift = int(lt / 10)  # 10분 단위 데이터

                    lead_col = target_data.shift(-shift, fill_value=target_data.iloc[-1])
                    leadtime_data_list.append(lead_col)

                leadtime_dataset = pd.concat(leadtime_data_list, axis=1)
                leadtime_dataset.columns = leadtime_header

                if not leadtime_dataset.empty:
                    leadtime_dataset.to_csv(os.path.join(save_path, f'alpha_1.0_target_leadtime.csv'), index=False)

            else:
                wl_col = f'WL_{WL_point[i-1]}'
                waterlevel = pd.concat([waterlevel, tmp_df[[wl_col]]], axis=1)
                first_val = float(tmp_df[wl_col].iloc[0])
                if first_val in [-99.0, -99.99]:
                    print(f'[Extreme Flood Scenario Creation] no WL -> convert to waterlevel from discharge at {wl_col}')
                    waterlevel[wl_col] = tmp_df['Forecast_outflow'].astype(float)

        if waterlevel.empty:            
            waterlevel = pd.DataFrame({'Date': measure_precipitation['Date']})
        
        waterlevel['Date'] = waterlevel['Date'].str.replace('@', ' ').str.replace('u', ' ')
        
        if not waterlevel.drop(columns=["Date"]).dropna(how="all").empty: 
            waterlevel.to_csv(os.path.join(save_path, 'alpha_1.0_waterlevel.csv'), index=False)

        # alpha 

        for j in range(len(alpha)):
            print('---------------- Alpha = {}'.format(alpha[j]))
            waterlevel = pd.DataFrame()
            alpha_folder = os.path.join(scenario_model_folder_path, scenario_name, 'Alpha={:.3f}'.format(alpha[j]), 'Output')

            for i, wl_file in enumerate(wl_post_file_list):
                file_path = os.path.join(alpha_folder, wl_file)

                if not os.path.exists(file_path):
                    print(f"[Extreme Flood Scenario Creation] File not found : {file_path}")
                    continue
            
                with open(file_path, 'r', encoding='cp949') as f:
                    tmp_lines = f.readlines()

                skip_index = next((k for k, line in enumerate(tmp_lines) if '------------------------------------' in line), None)
                if skip_index is None:
                    print(f"[Extreme Flood Scenario Creation] {wl_file} does not contain data separator.")
                    continue

                tmp_lines = tmp_lines[skip_index + 1:]
                data_split = [x.strip().split() for x in tmp_lines]
                tmp_df = pd.DataFrame(data_split)

                if len(tmp_df.columns) == 9:
                    if i == 0:
                        tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                        'Measure_discharge', f'Target_{target_point[i]}', 'Measure_stage', 'Remarks']
                    else:
                        tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                        'Measure_discharge', f'WL_{WL_point[i-1]}', 'Measure_stage', 'Remarks']
                elif len(tmp_df.columns) == 8:
                    if i == 0:
                        tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                        'Measure_discharge', f'Target_{target_point[i]}', 'Measure_stage']
                    else:
                        tmp_df.columns = ['Date', 'Avg_rainfall', 'Cum_rainfall', 'Forecast_inflow', 'Forecast_outflow',
                                        'Measure_discharge', f'WL_{WL_point[i-1]}', 'Measure_stage']
                else:
                    print(f"[Extreme Flood Scenario Creation] {wl_file} has unexpected number of columns: {len(tmp_df.columns)}")
                    continue

                if i == 0:
                    waterlevel = pd.concat([waterlevel, tmp_df[['Date', f'Target_{target_point[i]}']]], axis=1)
                    target_data = waterlevel.iloc[:, 1].astype(float).reset_index(drop=True)

                    # Leadtime 생성
                    leadtime_data_list = []
                    for lt in Leadtime:
                        if "min" in lead_cols[0]:
                            shift = int(lt / 10)
                        elif "H" in lead_cols[0]:
                            shift = int((lt * 60) / 10)
                        else:#없을경우 min으로 간주
                            shift = int(lt / 10)

                        lead_col = target_data.shift(-shift, fill_value=target_data.iloc[-1])
                        leadtime_data_list.append(lead_col)

                    leadtime_dataset = pd.concat(leadtime_data_list, axis=1)
                    leadtime_dataset.columns = leadtime_header

                    if not leadtime_dataset.empty:
                        leadtime_dataset.to_csv(os.path.join(save_path, f'alpha_{alpha[j]}_target_leadtime.csv'), index=False)

                    target_leadtime_total = pd.concat([target_leadtime_total, leadtime_dataset], ignore_index=True)

                else:
                    wl_col = f'WL_{WL_point[i-1]}'
                    waterlevel = pd.concat([waterlevel, tmp_df[[wl_col]]], axis=1)
                    first_val = float(tmp_df[wl_col].iloc[0])
                    if first_val in [-99.0, -99.99]:
                        print(f'[Extreme Flood Scenario Creation] no WL -> convert to waterlevel from discharge at {wl_col}')     
                        waterlevel[wl_col] = tmp_df['Forecast_outflow'].astype(float)
                        if j == 1:
                            no_WL_list.append(wl_col)

            if waterlevel.empty:            
                waterlevel = pd.DataFrame({'Date': measure_precipitation['Date']})

            waterlevel['Date'] = waterlevel['Date'].str.replace('@', ' ').str.replace('u', ' ')

            if not waterlevel.drop(columns=["Date"]).dropna(how="all").empty: 
                waterlevel.to_csv(os.path.join(save_path, 'alpha_{:.3f}_waterlevel.csv'.format(alpha[j])), index=False)
            
            waterlevel_total = pd.concat([waterlevel_total, waterlevel], ignore_index=True)

        # save        
        if not waterlevel_total.drop(columns=["Date"]).dropna(how="all").empty: 
            waterlevel_total.to_csv(os.path.join(save_path, 'total_waterlevel.csv'), index=False)        

        waterlevel_correction = waterlevel_total   # 수위 변환을 위해 미리 만들어둠
        
        if not target_leadtime_total.empty: 
            target_leadtime_total.to_csv(os.path.join(save_path, 'total_target_leadtime.csv'), index=False)

        # 수치형 변환
        for col in waterlevel_total.columns[1:]:
            waterlevel_total[col] = waterlevel_total[col].astype(float)

        print("[Extreme Flood Scenario Creation] No WL points :", no_WL_list)
                
        # no_WL_list
        # make the WL using rating curve   Q = a * (h - b) ^c

        if len(no_WL_list) != 0:

            hq_path = os.path.join(modelObservation_folder, 'H_Q_curve.csv')

            try:
                h_q_curve_list = pd.read_csv(hq_path)
            except Exception as e:
                # 파일 없음 → 모든 관측소 -99.0
                for obs in no_WL_list:
                    wl_values = np.array(waterlevel_correction.get(obs, []))
                    wl_values[:] = -99.0
                    waterlevel_correction[obs] = wl_values.tolist()
                    print(f"[Extreme Flood Scenario Creation] H_Q_curve.csv file not found : {obs} set to -99.0")
            else:
                for n, obs in enumerate(no_WL_list):
                    wl_values = np.array(waterlevel_correction.get(obs, []))

                    try:
                        if wl_values.size == 0:
                            wl_values = np.array([-99.0])
                            raise ValueError("Empty waterlevel_correction value")

                        station_code = int(obs[-7:])
                        tmp_hq = h_q_curve_list[h_q_curve_list['obs_code'] == station_code].sort_values('phase')

                        if tmp_hq.empty:
                            wl_values[:] = -99.0
                            print(f"[Extreme Flood Scenario Creation] obs_code {station_code} not found : {obs} set to -99.0")
                        else:
                            a = tmp_hq['a'].values
                            b = tmp_hq['b'].values
                            c = tmp_hq['c'].values
                            range_q = tmp_hq['range_Q'].values

                            if len(a) != len(range_q):
                                raise ValueError("Length of range_q does not match number of coefficients")

                            for j in range(len(a)):
                                if j == 0:
                                    mask = (wl_values >= 0) & (wl_values < range_q[j])
                                elif j == len(a) - 1:
                                    mask = wl_values >= range_q[j - 1]
                                else:
                                    mask = (wl_values >= range_q[j - 1]) & (wl_values < range_q[j])

                                wl_values[mask] = np.round((wl_values[mask] / a[j])**(1/c[j]) + b[j], 2)

                    except Exception as e:
                        wl_values[:] = -99.0
                        print(f"[Extreme Flood Scenario Creation] Error occurred while converting {obs} → set to -99.0 ({e})")

                    waterlevel_correction[obs] = wl_values.tolist()
                        
        # save the waterlevel_correnction
        
        if not waterlevel_correction.drop(columns=["Date"]).dropna(how="all").empty: 
            waterlevel_correction.to_csv(os.path.join(save_path,'total_waterlevel_correction.csv'),index=False) 

        # Make Scenario Dataset (+Predict_WS)
        scenario_dataset = pd.concat([waterlevel_correction,
                                      RF_cum_total.iloc[:, RF_cum_total.columns.str.startswith("RF_") & ~RF_cum_total.columns.str.contains("RF_cum")],
                                      DI_Dataset_total.iloc[:,range(1,len(DI_Dataset_total.columns))],
                                      REL_Dataset_total.iloc[:,range(1,len(REL_Dataset_total.columns))],
                                      discharge_total.iloc[:,range(1,len(discharge_total.columns))],                                      
                                      TIDE_Dataset_total.iloc[:,range(1,len(TIDE_Dataset_total.columns))],
                                      WS_Dataset_total.iloc[:,WS_Dataset_total.columns.str.startswith("WS_") & ~WS_Dataset_total.columns.str.contains("Predict_WS")],
                                      RF_cum_total.iloc[:, RF_cum_total.columns.str.startswith("RF_cum")],
                                      WS_Dataset_total.iloc[:,WS_Dataset_total.columns.str.contains("Predict_WS")],
                                      target_leadtime_total],
                                      axis=1)
        
        scenario_dataset['Date'] = pd.to_datetime(scenario_dataset['Date'], errors='coerce')
        #scenario_dataset['Date'] = scenario_dataset['Date'].dt.strftime("%Y-%m-%d %H:%M:00")
        scenario_dataset['Date'] = scenario_dataset['Date'].dt.floor('min')

        # save Scenario Dataset (only_scenario)
        dataset_scenario_path = os.path.join(save_path,'Scenario_Dataset_Target_{}_filtered_delete_null.csv'.format(target_point[0]))
        scenario_dataset.to_csv(dataset_scenario_path, index=False)        

        # save Original + Scenario Dataset
        if os.path.exists(dataset_origin_path) and os.path.exists(dataset_scenario_path):

            # read_csv_file
            df1 = pd.read_csv(dataset_origin_path, encoding="utf-8-sig")
            df2 = pd.read_csv(dataset_scenario_path, encoding="utf-8-sig")

            # Date 컬럼을 datetime으로 변환
            df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
            df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

            # 초 부분을 항상 00으로 맞추기
            df1['Date'] = df1['Date'].dt.floor('min')
            df2['Date'] = df2['Date'].dt.floor('min')

            if set(df1.columns) == set(df2.columns):
                df2_reordered = df2[df1.columns]

                result = pd.concat([df2_reordered, df1, df2_reordered], ignore_index=True)
                save_file = os.path.join(save_path, f'Dataset_Target_{target_point[0]}_filtered_delete_null.csv')
                result.to_csv(save_file, index=False, encoding="utf-8-sig")          


        # save Merge Scenario Dataset
        return True, "Scenario creation completed."
