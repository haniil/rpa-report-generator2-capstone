# rpa-report-generator-capstone
report generator 3 
2023.12.05.

프로젝트명 
펀드 운용 보고서 자동화

1. 기간별 수익률 
펀드 : 8186 수정기준가 - (수정기준가 - 수정기준가iloc[:0]) / 수정기준가iloc[:0]
KOSPI : 8186 KOSPI지수 
KOSPI200 : 8186 KOSPI200지수
KOSDAQ : 8186 KOSDAQ지수
S&P500 : dataset-price-SPX

펀드의 기간에 따라 데이터프레임을 다르게 조정
2년을 넘지 못하면 1년까지만 나오게 하고 3년을 넘긴다면 3년도 나오게 하는 기능


2. 성능 평가 지표 
누적수익률 : 8186 수정기준가 
연환산 수익률 : 기간수익률 *365/ 펀드 설정일 이후 현재까지 경과일수(기간수익률을 1년 단위로 환산한 수익률)= 누적수익률*365/날짜차이계산
그러나 다른 보고서에서는 값이 다름

변동성 : 변동성은 일일 수익률의 표준편차*(연간거래일수)**2 
이때 연간거래일수를 365로 계산 
그러나 연간거래일수는 일반적으로 252일로 계산 

샤프비율 : 연환산 수익률 - 무위험 수익률 /연환산 변동성
보고서의 샤프비율 어떻게 구했는지 확인 필요

Winning Ratio : 8186 전일대비 BM수익률과 다른 지수의 전일대비 차가 0인 날짜들은 쉬는 날이라 판단하고 이 날짜들을 필터링해야 하는 과정필요할듯 
거래 수가 아닌 날짜를 기준으로 함
이익 거래일 / 전체 거래일 * 100%

MDD : MAX(최대치-최소치)/최대치
여기서 최대치는 기간중 최대 펀드 가치
최소치는 그 이후의 최소 펀드가치

3. 월간 수익률 
펀드 : 8186
KOSPI : 8186
초과수익 : 펀드 - KOSPI

4. 펀드와 각종 지수에 대해 수익률 그래프 그리기 