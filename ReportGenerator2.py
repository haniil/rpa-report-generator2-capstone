import pandas as pd
import numpy as np
import os 
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import re
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import DateOffset
from ShiningPebbles import * 
from functools import reduce
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly

# 기본 메서드
    
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

def save_df_to_file(df, file_folder, subject, currency, file_memo, file_code,input_date, include_index=True, file_extension='.csv', archive=False, archive_folder='./archive'):
    def get_today(form='%Y%m%d'):
        return datetime.now().strftime(form)
    try:
        os.makedirs(file_folder, exist_ok=True)

        save_time = get_today()
        file_name = f'dataset-{subject}-{currency}-{file_memo}-code{file_code}-date{input_date}-save{save_time}{file_extension}'
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

def save_plot_to_HTML(fig, subject, fund_code, currency, input_date, file_folder='./plots',file_extension='.html'):
    def get_today(form='%Y%m%d'):
        return datetime.now().strftime(form)

    try:
        os.makedirs(file_folder, exist_ok=True)
        save_time = get_today()
        file_name = f'plot-{subject}-{currency}-code{fund_code}-date{input_date}-save{save_time}{file_extension}'
        file_path = os.path.join(file_folder, file_name)

        plotly.offline.plot(fig, filename=file_path, auto_open=False)

        print(f'Plot saved as {file_path}')
    except Exception as e:
        print(f"Error: {e}")

class Index:
    def __init__(self):
        
        self.df = None
        self.folder_path = './dataset-index'
                # date 관련 컬럼명
        self.date_column = '일자'
        self.index_names = ['CCMP', 'KOSDAQ', 'KOSPI', 'KOSPI200','MSCI', 'RUSSELL1000', 'SPX', 'MXKR']
        self.bbg_index_names = ['CCMP', 'KOSDAQ', 'KOSPI', 'KOSPI2','MSCI', 'RAY', 'SPX','MXKR']

        # index_df
        self.df_ccmp = None
        self.df_kosdaq = None
        self.df_kospi = None
        self.df_kospi200 = None
        self.df_msci = None
        self.df_russell1000 = None
        self.df_spx = None
        self.df_mxkr = None

        self.open_df_bbg_index_raw()

    def open_df_bbg_index_raw(self):
        file_list = scan_files_including_regex(file_folder=self.folder_path, regex=r'dataset-price-', option='path')

        # 인덱스 이름과 Bloomberg 인덱스 이름 매핑
        index_mapping = dict(zip(self.index_names, self.bbg_index_names))

        for file_path in file_list:
            index_name = re.search(r'dataset-price-(\w+).csv', file_path).group(1)
            
            # 추출된 인덱스 이름이 매핑에 존재하는지 확인
            if index_name in index_mapping:
                # 해당 Bloomberg 인덱스 이름 가져오기
                bbg_name = index_mapping[index_name]

                df = pd.read_csv(file_path)

                # 'price' 칼럼을 적절한 인덱스 이름으로 변경
                if 'price' in df.columns:
                    df.rename(columns={'price': f'{bbg_name} INDEX'}, inplace=True)

                # 'ticker' 칼럼이 없는 경우 'date' 칼럼을 'ticker'로 변경
                if 'ticker' not in df.columns and 'date' in df.columns:
                    df.rename(columns={'date': 'ticker'}, inplace=True)

                # 숫자 변환 및 결측치 처리
                df[f'{bbg_name} INDEX'] = pd.to_numeric(df[f'{bbg_name} INDEX'], errors='coerce')
                df = df.dropna(subset=[f'{bbg_name} INDEX']).reset_index(drop=True)
                df.rename(columns={f'{bbg_name} INDEX': index_name.upper(), 'ticker': self.date_column}, inplace=True)

                # 특정 인덱스 이름에 대한 추가 처리
                if index_name.upper() == 'SPX':
                    df.rename(columns={'SPX': 'S&P 500'}, inplace=True)
                elif index_name.upper() == 'MXKR':
                    df.rename(columns={'MXKR': 'MSCI KR'}, inplace=True)

                setattr(self, f'df_{index_name.lower()}', df)

        return df
    
class M8186(Index):
    def __init__(self, fund_code, start_date =None, end_date = None, menu_code = '8186', currency = 'KRW', 
                 option = ['KOSPI', 'KOSPI200', 'KOSDAQ', 'S&P 500', 'MSCI', 'RUSSELL1000', 'CCMP', 'MSCI KR']):
        super().__init__()
        self.fund_code = fund_code
        self.menu_code = menu_code
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.df = None  # 데이터프레임을 위한 초기화
        self.columns_multiindex = ['수정기준가'] + option
        self.columns_singleindex = self.columns_multiindex[:2]

        self.fund_name = self.get_fund_name()

    def open_df_raw(self):
        lst = scan_files_including_regex(file_folder = './캡스톤데이터2', regex = f'menu{self.menu_code}-code{self.fund_code}')
        lst = sorted(lst, reverse = True)
        file_path = lst[0]
        full_path = os.path.join(os.getcwd(), '캡스톤데이터2', file_path)
        df = pd.read_csv(full_path)

        if df.isnull().all(axis=1).any():  # 데이터프레임에 누락된 값이 있는지 확인
            raise ValueError("데이터 파일에 누락된 부분이 존재합니다. 데이터 확인이 필요합니다.")
        return df
    
    def get_df_ref(self, columns=None):
        self.df = self.open_df_raw()
        default_columns = ['일자', '수정기준가','CD91']

        # 전달된 칼럼 리스트가 없으면 기본 칼럼 리스트 사용
        if columns is None:
            columns = default_columns

        # 선택된 칼럼만 데이터프레임에 적용
        self.df = self.df[columns]

        if self.start_date is None:
            self.start_date = self.df['일자'].min()

        if self.end_date is None:
            self.end_date = self.df['일자'].max()

        return self.df

    def get_merged_df(self, avoid_nan=True):
        self.open_df_bbg_index_raw()
        df_ref = self.get_df_ref()
        df_ref['일자'] = pd.to_datetime(df_ref['일자'])

        new_date = df_ref['일자'].iloc[0] - timedelta(days=1)

        # 새로운 행에 대한 딕셔너리 생성 (일자는 new_date, 나머지 칼럼은 None으로 설정)
        new_row_data = {col: [None] if col != '일자' else [new_date] for col in df_ref.columns}

        # 새로운 행을 데이터프레임으로 변환
        new_row = pd.DataFrame(new_row_data)

        # 새로운 행을 기존 데이터프레임에 추가
        df_ref = pd.concat([new_row, df_ref]).reset_index(drop=True)

        # 병합할 데이터프레임 리스트 생성
        dataframes = [df_ref, self.df_ccmp, self.df_kosdaq, self.df_kospi, self.df_kospi200, self.df_msci, self.df_russell1000, self.df_spx, self.df_mxkr]
        for df in dataframes:
            if df['일자'].dtype == 'object':
                df['일자'] = pd.to_datetime(df['일자'])
        # reduce를 사용하여 여러 데이터프레임 병합
        self.df = reduce(lambda left, right: pd.merge(left, right, on='일자', how='left'), dataframes)

        # 비어 있는 값들을 각 열의 바로 앞 행의 값으로 대체
        if avoid_nan:
            self.df.ffill(inplace=True)

        # 첫 번째 행의 0 또는 NaN 값을 대체
        for column in self.df.columns:
            if self.df[column].iloc[0] == 0 or pd.isna(self.df[column].iloc[0]):
                self.df.at[0, column] = self.df[column].iloc[1]

        self.df = self.df[['일자'] + self.columns_multiindex]

        return self.df 
    
    def get_exchange_rate_data(self):
        self.start_date = pd.to_datetime(self.start_date) - pd.Timedelta(days=1)
        # USD/KRW의 역사적 환율 데이터를 가져옵니다
        exchange_rate_df = fdr.DataReader('USD/KRW', self.start_date, self.end_date)
        # open과 close의 중간값을 기준환율로 계산합니다
        exchange_rate_df['Standard_Rate'] = (exchange_rate_df['Open'] + exchange_rate_df['Close']) / 2
        return exchange_rate_df

    def get_exchage_rate_df(self):
        self.get_merged_df()
        self.filter_by_date_range()
        self.convert_to_float()
        self.fill_zero_with_previous()

        # 환율 데이터 가져오기
        exchange_rate_df = self.get_exchange_rate_data()
        self.df['일자'] = pd.to_datetime(self.df['일자'])
        exchange_rate_df['일자'] = exchange_rate_df.index
        exchange_rate_df['일자'] = pd.to_datetime(exchange_rate_df['일자'])
        self.df = pd.merge(self.df, exchange_rate_df[['일자', 'Standard_Rate']], on='일자', how='left')
        
        # USD가 아닐 경우 KRW 컬럼에 환율 적용
        if self.currency == 'USD':
            krw_columns = ['수정기준가', 'KOSPI', 'KOSPI200', 'KOSDAQ']
            for column in krw_columns:
                if column in self.df.columns:
                    self.df[column] = self.df[column] / self.df['Standard_Rate']
        else:
            # USD로 표시되는 칼럼들을 찾아 환율 적용
            krw_columns = set(['수정기준가', 'KOSPI', 'KOSPI200', 'KOSDAQ'])
            usd_columns = set(self.columns_multiindex) - krw_columns
            for column in usd_columns:
                if column in self.df.columns:
                    self.df[column] = self.df[column] * self.df['Standard_Rate']

        return self.df
    
    def fill_zero_with_previous(self, columns=None):
        if columns is None:
            columns = self.columns_multiindex
        
        for column in columns:
            self.df[column] = self.df[column].replace(0, None)
            self.df[column] = self.df[column].ffill()
        return self.df

    def convert_to_float(self, columns=None):
        if columns is None:
            columns = self.columns_multiindex

        for column in columns:
            self.df[column] = self.df[column].apply(lambda x: float(x.replace(',', '' )) if isinstance (x,str) else x)
        return self.df 
            
    def filter_by_date_range(self):
        # '일자' 컬럼을 datetime 타입으로 변환
        self.df['일자'] = pd.to_datetime(self.df['일자'])
        self.start_date = pd.to_datetime(self.start_date)

        # start_date와 end_date를 기준으로 데이터 필터링
        self.df = self.df[(self.df['일자'] >= (self.start_date - pd.Timedelta(days=1))) & (self.df['일자'] <= self.end_date)]

        return self.df

    def get_fund_name(self):
        self.df = self.open_df_raw()
        value  = self.df['펀드명'].iloc[0]
        fund_name = re.sub(r"\(.*?\)", "", value.replace(" ", ""))
        return fund_name

    def get_assets_under_management(self):
        self.df = self.open_df_raw()
        self.convert_to_float(['순자산'])
        if self.end_date is None:
            self.end_date = self.df['일자'].max()
        self.df = self.df[self.df['일자'] == self.end_date]
        value = self.df['순자산'].iloc[0]

        return value


    def calculate_cumulative_return_for_df(self, df, columns = None):
        df = df.copy()  # 명시적으로 데이터프레임 복사본 생성
        if columns is None:
            columns = self.columns_multiindex

        for column_name in columns:
            if column_name in df.columns:
                initial_value = df[column_name].iloc[0]
                if initial_value != 0:
                    updated_values = ((df[column_name] - initial_value) / initial_value) * 100
                else:
                    updated_values = df[column_name] * 0  # 초기값이 0일 경우, 결과는 모두 0
                updated_values.iloc[0] = 0  # 첫 번째 행의 수익률을 0으로 설정
                df.loc[:, column_name + ' (%)'] = updated_values
        return df   

    def get_cumulative_return(self):
        df = self.calculate_cumulative_return_for_df(self.df, columns = self.columns_singleindex)
        cumulative_returns = {}

        for column_name in self.columns_singleindex:
            cumulative_return = df[column_name + ' (%)'].iloc[-1]
            cumulative_returns[column_name] = cumulative_return

        return cumulative_returns
    
    def get_annualized_return(self):
        df = self.calculate_cumulative_return_for_df(self.df, self.columns_singleindex)
        start_date = df['일자'].iloc[0]
        end_date = df['일자'].iloc[-1]
        days = (end_date - start_date).days 

        # 연환산 수익률 계산을 위한 딕셔너리 초기화
        annualized_returns = {}

        for column_name in self.columns_singleindex:
            cumulative_return = df[column_name + ' (%)'].iloc[-1]
            annualized_return = (cumulative_return * 365) / days
            annualized_returns[column_name] = annualized_return
            
        return annualized_returns

    def get_volatility(self):
        # 변동성 계산을 위한 딕셔너리 초기화
        volatility = {}

        self.daily_returns = self.df[self.columns_singleindex].pct_change()
        self.daily_returns.iloc[0] = 0
        self.daily_returns_std = self.daily_returns.std()

        # 각 칼럼에 대한 변동성 계산
        for column_name in self.columns_singleindex:
            # volatility[column_name] = self.daily_returns_std[column_name] * (365 ** 0.5) * 100
            volatility[column_name] = self.daily_returns_std[column_name] * (370 ** 0.5) * 100
        return volatility
    
    def get_risk_free_rate(self):
        # CD91 데이터 불러오기
        df_cd91 = self.open_df_raw()
        df_cd91['일자'] = pd.to_datetime(df_cd91['일자'])
        df_cd91 = df_cd91[(df_cd91['일자'] >= self.start_date) & (df_cd91['일자'] <= self.end_date)]
        # CD91 평균 수익률을 무위험 수익률로 계산
        risk_free_rate = df_cd91['CD91'].mean() / 100 / 365
        return risk_free_rate

    def get_sharpe_ratio(self):
        risk_free_rate = self.get_risk_free_rate()
        # 일일 수익률 계산
        self.daily_returns = self.df[self.columns_singleindex].pct_change()
        self.daily_returns.iloc[0] = 0

        # 일일 수익률의 평균과 표준편차 계산
        self.daily_returns_mean = self.daily_returns.mean()
        self.daily_returns_std = self.daily_returns.std()

        # Sharpe 비율 계산을 위한 딕셔너리 초기화
        sharpe_ratios = {}

        # 각 칼럼에 대한 Sharpe 비율 계산
        for column_name in self.columns_singleindex:
            mean_return = self.daily_returns_mean[column_name]
            std_return = self.daily_returns_std[column_name]

            if std_return != 0:
                # sharpe_ratios[column_name] = (mean_return - risk_free_rate) / std_return * np.sqrt(365)
                sharpe_ratios[column_name] = (mean_return - risk_free_rate) / std_return * np.sqrt(370)
            else:
                sharpe_ratios[column_name] = None  # 표준편차가 0이면 Sharpe 비율을 계산할 수 없음

        return sharpe_ratios 

    def get_winning_ratio(self):
        # 수정기준가에서 비교 대상 지수를 뺀 값 계산
        df = self.df.copy()

        # '일자' 칼럼을 datetime 타입으로 변환
        df['일자'] = pd.to_datetime(df['일자'])

        # self.end_date로부터 1년 전의 날짜 계산
        # one_year_ago = pd.to_datetime(self.end_date) - pd.DateOffset(years=1)
        one_year_ago = pd.to_datetime(self.end_date) - pd.Timedelta(days=370)
        # # self.end_date로부터 1년 전까지의 데이터만 필터링
        df = df[df['일자'] >= one_year_ago]

        # self.columns_singleindex에 따라 데이터 처리
        fund_column = self.columns_singleindex[0]  # '수정기준가'
        index_column = self.columns_singleindex[1]  # 'kospi' 또는 'spx'

        df[fund_column] = df[fund_column].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
        df[index_column] = df[index_column].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
        df[[fund_column, index_column]] = df[[fund_column, index_column]].pct_change()
        df.iloc[0] = 0
        diff = df[fund_column] - df[index_column]
        df['diff'] = diff.round(3) 
        # df['diff'] = diff
        # 양수인 값과 음수인 값의 개수 계산
        positive_count = (df['diff'] >= 0).sum()
        negative_count = (df['diff'] < 0).sum()

        # 전체 비율 계산
        total_count = positive_count + negative_count
        fund_winning_ratio = positive_count / total_count * 100 if total_count != 0 else 0
        index_winning_ratio = negative_count / total_count * 100 if total_count != 0 else 0
        fund_winning_ratio = round(fund_winning_ratio, 1)
        index_winning_ratio = round(index_winning_ratio, 1)
        # 결과 저장
        result = {
            fund_column: fund_winning_ratio,
            index_column: index_winning_ratio
        }

        return result

    def get_mdd(self):
        # MDD를 저장할 딕셔너리 초기화
        mdd = {}

        # columns_singleindex 내의 각 칼럼에 대해 MDD 계산
        for column in self.columns_singleindex:
            mdd_values = []
            for index, max_value in self.df[column].items():
                min_value_after = self.df.loc[index:, column].min()
                current_mdd = (max_value - min_value_after) / max_value if max_value > 0 else 0
                mdd_values.append(current_mdd)
  
            # 최대 MDD 값 계산
            max_mdd = max(mdd_values) * 100

            # 각 칼럼별 최대 MDD 저장
            mdd[column] = max_mdd

        return mdd

    def filter_for_period(self, months):
        if months is not None:
            # 현재 가장 최근 날짜를 구함
            df_end_date = self.df['일자'].max()

            # 지정된 개월 수만큼 과거 날짜를 계산
            period_start_date = df_end_date - DateOffset(months=months)

            # period_start_date보다 이전 데이터를 필터링
            filtered_df = self.df[self.df['일자'] >= period_start_date]

            return filtered_df
        else:
            # months가 None이면 전체 데이터프레임 반환
            return self.df


    def generate_period_df(self):
        # self.df의 최대 및 최소 날짜 찾기
        df_start_date = self.df['일자'].min()
        df_end_date = self.df['일자'].max()

        # 가능한 모든 기간을 검사하여 default_periods 설정
        potential_periods = [1, 3, 6, 12, 24, 36, 48, 60]
        default_periods = []

        for period in potential_periods:
            period_start_date = df_end_date - DateOffset(months=period)
            if period_start_date >= df_start_date:
                default_periods.append(period)

        period_dfs = {}  # 각 기간에 해당하는 데이터프레임을 저장할 딕셔너리

        # 각 기간에 대한 데이터프레임 생성
        for period in default_periods:
            period_dfs[f"{period}m"] = self.filter_for_period(period)

        # YTD 데이터프레임 생성
        # 현재 연도 필터링
        current_year = pd.Timestamp.now().year
        current_year_df = self.df[self.df['일자'].dt.year == current_year]
        period_dfs['YTD'] = current_year_df

        return period_dfs

    def format_period(self, period):
        """
        '기간' 값을 포맷하는 함수. 'usd'일 때는 영어로, 'krw'일 때는 한글로 반환합니다.
        예: '1m' -> '1 Month'/'1개월', '12m' -> '1 Year'/'1년' 등
        """
        try:
            months = int(period.replace('m', ''))
            if self.currency == 'USD':
                if months == 1:
                    return '1 Month'
                elif months < 12:
                    return f'{months} Months'
                elif months % 12 == 0:
                    years = months // 12
                    return f'{years} Year' if years == 1 else f'{years} Years'
            else:
                if months < 12:
                    return f'{months}개월'
                elif months % 12 == 0:
                    years = months // 12
                    return f'{years}년'
        except ValueError:
            return period

    def get_final_cumulative_returns(self, period_dfs):
        final_returns_data = []

        for period, df in period_dfs.items():
            formatted_period = self.format_period(period)
            last_row = df.iloc[-1]
            row_data = {'Period' if self.currency == 'USD' else '기간': formatted_period,
                        'Fund' if self.currency == 'USD' else '펀드': last_row.get('수정기준가 (%)', None)}

            # self.columns_multiindex를 사용하여 동적으로 데이터 처리
            for column in self.columns_multiindex:
                if column != '수정기준가':  # '수정기준가' 컬럼을 제외
                    column_key = f'{column} (%)'
                    if column_key in df.columns:
                        row_data[column] = last_row.get(column_key, None)

            final_returns_data.append(row_data)

        # 데이터를 기반으로 새로운 데이터프레임 생성
        final_returns_df = pd.DataFrame(final_returns_data)
        final_returns_df.set_index('Period' if self.currency == 'USD' else '기간', inplace=True)

        return final_returns_df
    
    def process_period_dfs(self):
        # 각 기간별 데이터프레임을 생성
        period_dfs = self.generate_period_df()

        # 각 데이터프레임에 대해 누적 수익률 계산
        for period, df in period_dfs.items():
            period_dfs[period] = self.calculate_cumulative_return_for_df(df)

        # 전체 기간에 대한 누적수익률 추가
        since_inception_label = 'Since Inception' if self.currency == 'USD' else '설정이후'
        period_dfs[since_inception_label] = self.calculate_cumulative_return_for_df(self.df)
        
        # 각 기간별 누적수익률의 마지막 값으로 구성된 데이터프레임을 반환
        final_returns_df = self.get_final_cumulative_returns(period_dfs)

        return final_returns_df
    
    def get_investment_performance_df(self):
        # 각 메서드를 호출하여 지표값을 가져옴
        cumulative_returns = self.get_cumulative_return()
        annualized_returns = self.get_annualized_return()
        volatility = self.get_volatility()
        sharpe_ratios = self.get_sharpe_ratio()
        winning_ratios = self.get_winning_ratio()
        mdd = self.get_mdd()

        # 데이터프레임 생성
        # currency 'usd'일 경우 영어 레이블 사용
        if self.currency == 'USD':
            summary_df = pd.DataFrame({
                'Cumulative Return': cumulative_returns,
                'Annualized Return': annualized_returns,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratios,
                'Winning Ratio': winning_ratios,
                'MDD': mdd
            })
        else:
            # 기본값인 'krw'일 경우 한글 레이블 사용
            summary_df = pd.DataFrame({
                '누적수익률': cumulative_returns,
                '연환산 수익률': annualized_returns,
                '변동성': volatility,
                '샤프비율': sharpe_ratios,
                'Winning Ratio': winning_ratios,
                'MDD': mdd
            })

        # columns_singleindex를 사용하여 행 인덱스 이름 변경
        index_rename_map = {
            self.columns_singleindex[0]: 'Fund' if self.currency == 'USD' else '펀드',
            self.columns_singleindex[1]: self.columns_singleindex[1]
        }
        summary_df = summary_df.rename(index=index_rename_map)

        return summary_df

    def get_monthly_dates(self):
        # '일자' 컬럼을 datetime 타입으로 변환
        self.df['일자'] = pd.to_datetime(self.df['일자'])

        # 각 달의 마지막날을 찾음
        monthly_last_dates = self.df.groupby(self.df['일자'].dt.to_period('M')).agg({'일자': 'last'}).reset_index(drop=True)

        # 데이터의 첫 값 추가
        first_date = self.df['일자'].iloc[0]
        monthly_dates = pd.concat([pd.Series([first_date]), monthly_last_dates['일자']], ignore_index=True)

        # '일자' 컬럼에서 monthly_dates에 해당하는 값만 필터링
        filtered_df = self.df[self.df['일자'].isin(monthly_dates)]

        return filtered_df
    
    def calculate_monthly_returns(self, df):
        # 월간 수익률 계산
        monthly_returns = df[self.columns_singleindex].pct_change().dropna() *100 

        # '일자' 칼럼의 날짜 형식을 '연-월' 형식으로 변경
        if not pd.api.types.is_datetime64_any_dtype(df['일자']):
            df['일자'] = pd.to_datetime(df['일자'])
        monthly_returns['일자'] = df['일자'].dt.strftime('%Y-%m')

        return monthly_returns
        

    def calculate_excess_return(self, monthly_returns, ytd_values):
        fund_column = self.columns_singleindex[0]  # '수정기준가'
        BM_column = self.columns_singleindex[1]  # 'KOSPI지수' 또는 'S&P 500'

        # 월간 초과수익률 계산
        excess_return_label = 'Excess Return' if self.currency == 'USD' else '초과수익'
        monthly_returns[excess_return_label] = monthly_returns[fund_column] - monthly_returns[BM_column]

        # YTD 초과수익률 계산
        for year, values in ytd_values.items():
            ytd_values[year][excess_return_label] = values[fund_column] - values[BM_column]

        return monthly_returns, ytd_values
    
    def calculate_ytd_values(self):
        ytd_values = {}
        monthly_dates_df = self.df

        if not monthly_dates_df.empty:  # 데이터프레임이 비어 있지 않은 경우에만 계산 수행
            for year in monthly_dates_df['일자'].dt.year.unique():
                year_data = monthly_dates_df[monthly_dates_df['일자'].dt.year == year]
                cumulative_year_data = self.calculate_cumulative_return_for_df(year_data, self.columns_singleindex) 
                ytd_values[year] = {column: cumulative_year_data[column + ' (%)'].iloc[-1] for column in self.columns_singleindex}
        return ytd_values
    
    def create_monthly_calendar_df(self, monthly_returns, ytd_values):
        df_list = []
        months_eng = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']

        # 'excess_return_label' 재정의
        excess_return_label = 'Excess Return' if self.currency == 'USD' else '초과수익'

        # 모든 연도를 수집하고 오름차순으로 정렬
        all_years = set(monthly_returns['일자'].str.slice(0, 4)).union(set(map(str, ytd_values.keys())))
        sorted_years = sorted(all_years)

        # 연도와 지표에 따라 데이터프레임을 구성합니다.
        for year in sorted_years:
            for indicator in self.columns_singleindex + [excess_return_label]:
                row = {'Year' if self.currency == 'USD' else '연도': year, 
                       'Indicator' if self.currency == 'USD' else '지표': indicator}
                for month in range(1, 13):
                    month_label = months_eng[month-1] if self.currency == 'USD' else f'{month}월'
                    monthly_value = monthly_returns[(monthly_returns['일자'].str.startswith(year)) & (monthly_returns['일자'].str.endswith(f'-{str(month).zfill(2)}'))].get(indicator)
                    row[month_label] = monthly_value.iloc[0] if not monthly_value.empty else None
                row['YTD'] = ytd_values.get(int(year), {}).get(indicator, None)
                df_list.append(row)

        # 데이터프레임 생성
        calendar_df = pd.DataFrame(df_list)
        calendar_df.set_index(['Year' if self.currency == 'USD' else '연도', 
                               'Indicator' if self.currency == 'USD' else '지표'], inplace=True)
        # 인덱스 변경을 위한 딕셔너리 생성
        index_rename_dict = {
            self.columns_singleindex[0]: 'Fund' if self.currency == 'USD' else '펀드',
            self.columns_singleindex[1]: self.columns_singleindex[1]
        }
        calendar_df = calendar_df.rename(index=index_rename_dict, level='Indicator' if self.currency == 'USD' else '지표')

        return calendar_df

    def adjust_percent(self, columns): 
        for column_name in columns: 
            initial_value = self.df[column_name].iloc[0]
            self.df[column_name + ' (%)'] = ((self.df[column_name] - initial_value) / initial_value) * 100
            self.df.loc[0, column_name + ' (%)'] = 0

    def get_index_df(self):
        if self.currency == 'USD':
            # USD 옵션 선택 시, 환율 데이터 적용
            self.get_exchage_rate_df()
        else:
            # KRW 옵션 또는 기타 옵션 선택 시, 기존 데이터 사용
            self.get_merged_df()

        self.filter_by_date_range()
        self.convert_to_float()
        self.fill_zero_with_previous()
        self.adjust_percent(self.columns_multiindex)

        return self.df

    #기간별 수익률을 위한 메인 메서드 
    def period_cumulative_return(self):

        self.get_exchage_rate_df()
        self.filter_by_date_range()
        self.convert_to_float()
        self.fill_zero_with_previous()

        final_returns_df = self.process_period_dfs()
        final_returns_df = final_returns_df.round(2)

        return final_returns_df

    #성능평가지표를 위한 메인 메서드
    def investment_performance(self):

        self.get_exchage_rate_df()
        self.filter_by_date_range()
        self.convert_to_float(self.columns_singleindex)
        self.fill_zero_with_previous(self.columns_singleindex)
        self.calculate_cumulative_return_for_df(self.df, self.columns_singleindex)
        summary_df = self.get_investment_performance_df()
        summary_df = summary_df.round(2)
        return summary_df

    #월간 수익률을 위한 메인 메서드
    def monthly_return(self):

        self.get_exchage_rate_df()
        self.filter_by_date_range()
        self.convert_to_float(self.columns_singleindex)
        self.fill_zero_with_previous(self.columns_singleindex)
        # YTD 값 계산
        ytd_values = self.calculate_ytd_values()
        monthly_filtered_df = self.get_monthly_dates()

        monthly_returns = self.calculate_monthly_returns(monthly_filtered_df)
        # 초과수익률 계산
        monthly_returns, ytd_values = self.calculate_excess_return(monthly_returns, ytd_values)
        # 월별 및 연간 수익률 데이터프레임 생성
        final_df = self.create_monthly_calendar_df(monthly_returns, ytd_values)
        final_df = final_df.round(1)
        return final_df

    def save_df_period_cumlative_return(self):
        period_cumulative_return_df = self.period_cumulative_return()
        self.end_date = self.end_date.replace('-', '')
        save_df_to_file(df = period_cumulative_return_df, file_folder = './dataset',subject = 'period_return', currency = self.currency, file_memo = 'menu8186', file_code = self.fund_code, input_date = self.end_date)

    def save_df_investment_performance(self):
        investment_performance_df = self.investment_performance()
        self.end_date = self.end_date.replace('-', '')
        save_df_to_file(df = investment_performance_df, file_folder = './dataset',subject = 'investment_performance', currency = self.currency, file_memo = 'menu8186', file_code = self.fund_code, input_date = self.end_date)

    def save_df_monthly_return(self):
        monthly_return_df = self.monthly_return()
        self.end_date = self.end_date.replace('-', '')
        save_df_to_file(df = monthly_return_df, file_folder = './dataset',subject = 'monthly_return', currency = self.currency, file_memo = 'menu8186', file_code = self.fund_code, input_date = self.end_date)

    def save_df(self):
        self.save_df_period_cumlative_return()
        self.save_df_investment_performance()
        self.save_df_monthly_return()


class M2160:
    def __init__(self, fund_code, start_date =None, end_date = None, menu_code = '2160'):
        self.fund_code = fund_code
        self.menu_code = menu_code
        self.start_date = start_date
        self.end_date = end_date

    def open_df_raw(self):
        lst = scan_files_including_regex(file_folder = './캡스톤데이터2', regex = f'menu{self.menu_code}-code{self.fund_code}')
        lst = sorted(lst, reverse = True)
        file_path = lst[0]
        full_path = os.path.join(os.getcwd(), '캡스톤데이터2', file_path)
        df = pd.read_csv(full_path)

        if df.isnull().all(axis=1).any():  # 데이터프레임에 누락된 값이 있는지 확인
            raise ValueError("데이터 파일에 누락된 부분이 존재합니다. 데이터 확인이 필요합니다.")
        return df
    
    def get_df_ref(self, columns=None):
        self.df = self.open_df_raw()
        default_columns = ['일자', '편입비중']
        self.df = self.df.drop(self.df.index[0])

        # 전달된 칼럼 리스트가 없으면 기본 칼럼 리스트 사용
        if columns is None:
            columns = default_columns

        # 선택된 칼럼만 데이터프레임에 적용
        self.df = self.df[columns]

        if self.start_date is None:
            self.start_date = self.df['일자'].min()

        if self.end_date is None:
            self.end_date = self.df['일자'].max()

        return self.df

    def filter_by_date_range(self):
        # '일자' 컬럼을 datetime 타입으로 변환
        self.df['일자'] = pd.to_datetime(self.df['일자'])

        # start_date와 end_date를 기준으로 데이터 필터링
        self.df = self.df[(self.df['일자'] >= self.start_date) & (self.df['일자'] <= self.end_date)]
        return self.df

    def convert_to_float(self, columns=None):
        for column in columns:
            self.df[column] = self.df[column].apply(lambda x: float(x.replace(',', '' )) if isinstance (x,str) else x)
        return self.df 

    def get_df_2160(self):
        self.get_df_ref()
        self.filter_by_date_range()
        self.convert_to_float(['편입비중'])

        return self.df


class Perfomance:
    def __init__(self, fund_code,  start_date =None, end_date = None, currency = 'KRW'):
        self.fund_code = fund_code
        self.start_date = start_date
        self.end_date = end_date
        self.date_column='일자'
        self.currency = currency

        m8186 = M8186(fund_code=self.fund_code, start_date= self.start_date, end_date= self.end_date, currency = self.currency)
        m2160 = M2160(fund_code=self.fund_code, start_date= self.start_date, end_date= self.end_date)
        self.get_index_df = m8186.get_index_df()
        self.get_proportion_df = m2160.get_df_2160()
        self.get_performace_df = pd.merge(self.get_index_df, self.get_proportion_df, on='일자', how='inner')
        self.fund_name = m8186.fund_name

        if self.start_date is None:
            self.start_date = self.get_performace_df['일자'].min()

        if self.end_date is None:
            self.end_date = self.get_performace_df['일자'].max()

    def get_performance_plot(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        df = self.get_performace_df
        col_date =  self.date_column
        col_fund = '수정기준가 (%)'
        col_proportion = '편입비중'
        # 모든 (%)로 끝나는 칼럼을 찾아서 리스트로 저장
        percentage_columns = [col for col in df.columns if col.endswith(' (%)') and col != '수정기준가 (%)']


        stock_proportion_label = "<b>Stock Proportion</b>" if self.currency == 'USD' else "<b>주식비중</b>"
        fund_label = "<b>Fund</b>" if self.currency == 'USD' else f"<b>{self.fund_name}</b>"

        min_value_left = int(min([df[col].min() for col in percentage_columns] + [df[col_fund].min()])) - 10
        max_value_left = int(max([df[col].max() for col in percentage_columns] + [df[col_fund].max()])) + 10

        # 문자열을 실수형으로 변환한 후 최소값과 최대값을 계산
        min_value_right = min(float(value) for value in df[col_proportion])
        max_value_right = max(float(value) for value in df[col_proportion]) + 20


        # min_value_left가 짝수가 아니라면, 원래 값보다 작은 가장 가까운 짝수로 조정
        if min_value_left % 10 != 0:
            min_value_left  = min_value_left - (min_value_left % 10)

        # max_value_left가 짝수가 아니라면, 원래 값보다 큰 가장 가까운 짝수로 조정
        if max_value_left % 10 != 0:
            max_value_left  = max_value_left + (10 - max_value_left % 10)

        # min_value_right가 10의 배수가 아니라면, 원래 값보다 작은 가장 가까운 10의 배수로 조정
        if min_value_right % 10 != 0:
            min_value_right = min_value_right - (min_value_right % 10)

        # max_value_right가 10의 배수가 아니라면, 원래 값보다 큰 가장 가까운 10의 배수로 조정
        if max_value_right % 10 != 0:
            max_value_right = max_value_right + (10 - max_value_right % 10)

    
        fig.add_trace(
            go.Scatter(x=df[col_date], y=df[col_proportion], name=stock_proportion_label, fill='tozeroy', line=dict(color='#90c4f5', width=0), opacity=0.1),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df[col_date], y=df[col_fund], name=fund_label, line=dict(color='orange', width=3)),
            secondary_y=False,
        )

        # 각 칼럼에 대한 그래프 플롯
        for col in percentage_columns:
            formatted_col_name = col.replace(' (%)', '')  # (%) 부분을 제거하여 새로운 이름 생성
            fig.add_trace(
                go.Scatter(x=df[col_date], y=df[col], name=f"<b>{formatted_col_name}</b>", line=dict(width=1)),
                secondary_y=False,
            )


        # 최솟값과 최댓값 사이의 간격 계산
        interval = (max_value_left - min_value_left) / 11

        # 눈금 값과 라벨 생성
        tick_values = [min_value_left + i * int(interval) for i in range(11)]

        # 눈금 라벨 생성
        tick_labels = [f"{value:.2f}%" for value in tick_values]

        fig.update_yaxes(
            secondary_y=False,
            range=[min_value_left, max_value_left],
            tickvals=tick_values,  # 눈금 값 설정
            ticktext=tick_labels,  # 눈금 라벨 설정
            nticks=11  # nticks를 6으로 설정
        )

        right_tick_values = [min_value_right + i * ((max_value_right - min_value_right) / 6) for i in range(7)]
        right_tick_labels = [f"{value:.2f}%" for value in right_tick_values]

        # Adjusting y-axis (right) ticks
        fig.update_yaxes(
            secondary_y=True,
            range=[min_value_right, max_value_right],
            tickvals=right_tick_values,  # 눈금 값 설정
            ticktext=right_tick_labels,  # 눈금 라벨 설정
            nticks=7
        )
        # Adding a thicker horizontal line at 0% on the left y-axis
        fig.add_shape(
            type="line",
            x0=df[col_date].min(),
            y0=0,
            x1=df[col_date].max(),
            y1=0,
            line=dict(
                color="Gray",
                width=1
            ),
            secondary_y=False
        )

        start_date = self.start_date    
        end_date = self.end_date
        
        # 월말 날짜를 계산합니다.
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        monthly_dates_list = monthly_dates.tolist()

        # 시작 날짜가 월말이 아니면, 첫 값을 시작 날짜로 교체합니다.
        if start_date not in monthly_dates:
            monthly_dates_list[0] = start_date

        # 종료 날짜가 월말이 아니면, 리스트에 추가합니다.
        # if end_date not in monthly_dates:
        #     monthly_dates_list.append(end_date)

        # 월말 눈금을 설정합니다.
        tickvals = pd.to_datetime(monthly_dates_list)
        ticktext = [date.strftime('%Y-%m-%d') for date in tickvals]

        # x축 속성을 업데이트합니다.
        fig.update_xaxes(
            range=[start_date, end_date],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-60
        )
        # Adjusting the legend position and setting graph size
        fig.update_layout(
            legend=dict(font=dict(size=10), x=0.5, xanchor="center", y=1.3, yanchor="top", orientation="h"),
            width=1000,  # Graph width
            height=400,   # Graph height
            paper_bgcolor='white',  # 그래프 외부 배경색 설정
            plot_bgcolor='white'    # 그래프 내부 배경색 설정
        )

        # Remove horizontal and vertical gridlines for both x-axis and y-axes
        fig.update_xaxes(showgrid=False)  # Remove vertical gridlines
        fig.update_yaxes(showgrid=True, gridcolor = 'lightgrey', secondary_y=False)  # Remove horizontal gridlines for left y-axis
        fig.update_yaxes(showgrid=False, secondary_y=True)  # Remove horizontal gridlines for right y-axis

        return fig
    
    def save_plot(self):
        fig = self.get_performance_plot()  # 그래프 생성

        # self.end_date가 문자열인 경우, '-'를 제거합니다.
        if isinstance(self.end_date, str):
            formatted_end_date = self.end_date.replace('-', '')
        else:
            # self.end_date가 datetime 객체인 경우, 문자열로 포맷팅합니다.
            formatted_end_date = self.end_date.strftime('%Y%m%d')

        save_plot_to_HTML(fig=fig, subject='performance', fund_code=self.fund_code, currency=self.currency, input_date=formatted_end_date)  # 외부 함수 호출하여 저장
    