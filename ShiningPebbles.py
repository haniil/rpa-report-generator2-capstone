import pandas as pd
import numpy as np
import os 
from datetime import datetime, timedelta
import re

def get_today(form='%Y-%m-%d'):
    mapping = {
        '%Y%m%d': datetime.now().strftime("%Y%m%d"),
        'yyyymmdd': datetime.now().strftime("%Y%m%d"),
        '%Y-%m-%d': datetime.now().strftime("%Y-%m-%d"),
        'yyyy-mm-dd': datetime.now().strftime("%Y-%m-%d"),
        'datetime': datetime.now(),
        '%Y%m%d%H': datetime.now().strftime("%Y%m%d%H"),
    }
    today = mapping[form]
    return today 

def scan_files_including_regex(file_folder, regex, option='name'):
    with os.scandir(file_folder) as files:
        lst = [file.name for file in files if re.findall(regex, file.name)]
    
    mapping = {
        'name': lst,
        'path': [os.path.join(file_folder, file_name) for file_name in lst]
    }
    return mapping[option]

def format_date(date):
    date = date.replace('-', '')
    date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
    return date

def save_df_to_file(df, file_folder, file_memo, file_extension='.csv', archive=False, archive_folder='./archive'):
    def get_today(form='%Y%m%d'):
        return datetime.now().strftime(form)
    try:
        save_time = get_today()
        file_name = f'dataset-{file_memo}-save{save_time}{file_extension}'
        file_path = os.path.join(file_folder, file_name)
        if os.path.exists(file_path) and archive:
            df_archive = pd.read_csv(file_path)
            os.makedirs(archive_folder, exist_ok=True)
            archive_file_name = 'archive-' + file_name
            archive_file_path = os.path.join(archive_folder, archive_file_name)
            df_archive.to_csv(archive_file_path, index=False)
            print(f'Archived: {archive_file_path}')
        df.to_csv(file_path, index=False)
        print(f'Saved: {file_path}')
    except Exception as e:
        print(f"Error: {e}")



def filter_by_date_range(df, date_column, start_date=None, end_date=None):

    # '일자' 컬럼을 datetime 타입으로 변환, 변환할 수 없는 값들은 NaT로 설정
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # start_date와 end_date에 따라 조건을 설정하여 필터링
    if start_date is None and end_date is not None:
        df = df[df[date_column] <= end_date]
    elif start_date is not None and end_date is None:
        df = df[df[date_column] >= start_date]
    elif start_date is not None and end_date is not None:
        df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    
    # 행의 결측치를 ffill로 채움. 
    df.loc[:, date_column] = df[date_column].ffill()
    return df