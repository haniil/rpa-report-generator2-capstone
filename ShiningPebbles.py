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

def scan_files_including_regex(file_folder, regex, option='path'):
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

def save_df_to_file(df, file_folder, subject, file_memo, file_code,input_date, include_index=False, file_extension='.csv', archive=False, archive_folder='./archive'):
    def get_today(form='%Y%m%d'):
        return datetime.now().strftime(form)
    try:
#         new_folder_path = os.path.join(file_folder, new_folder_name)
#         os.makedirs(new_folder_path, exist_ok=True)
        save_time = get_today()
        file_name = f'dataset-{subject}-{file_memo}-code{file_code}-date{input_date}-save{save_time}{file_extension}'
        file_path = os.path.join(file_folder, file_name)
        if os.path.exists(file_path) and archive:
            df_archive = pd.read_csv(file_path)
            os.makedirs(archive_folder, exist_ok=True)
            archive_file_name = 'archive-' + file_name
            archive_file_path = os.path.join(archive_folder, archive_file_name)
            df_archive.to_csv(archive_file_path, index=False)
            print(f'Archived: {archive_file_path}')
        df.to_csv(file_path, index=include_index, encoding='utf-8-sig')
        print(f'Saved: {file_path}')
    except Exception as e:
        print(f"Error: {e}")
        