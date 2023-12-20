# rpa-report-generator-capstone
report generator 3 
2023.12.05.

프로젝트명 
펀드 운용 보고서 자동화

현재 가지고 있는 보고서의 기간은 2021.07.29 ~ 2023.10.31
현재 가지고 있는 데이터의 기간은 2021.07.29 ~ 2023.10.30
하루 차이가 존재하여 실제 보고서의 값과 오차가 존재
하루 차이가 생각보다 큰 오차를 발생

1. 기간별 수익률 period_cumulative_return
펀드 : 8186 수정기준가 - (수정기준가 - 수정기준가iloc[:0]) / 수정기준가iloc[:0]
KOSPI : 8186 KOSPI지수 
KOSPI200 : 8186 KOSPI200지수
KOSDAQ : 8186 KOSDAQ지수
S&P500 : dataset-price-SPX

펀드의 기간에 따라 데이터프레임을 다르게 조정
2년을 넘지 못하면 1년까지만 나오게 하고 3년을 넘긴다면 3년도 나오게 하는 기능

추가할 내용 
option = 'krw' 'usd' 에 따라 환율을 적용한 수익률
환율 적용시 질문점
기존환율 vs ask, bid
누적수익률에서의 적용

2. 성능 평가 지표  investment_performance
누적수익률 : 8186 수정기준가 

연환산 수익률 : 기간수익률 *365/ 펀드 설정일 이후 현재까지 경과일수(기간수익률을 1년 단위로 환산한 수익률)= 누적수익률*365/날짜차이계산 
그러나 다른 보고서에서는 값이 다름
설정되지 1년이 경과하지 않은 펀드는 연환산 수익률이 표시되지 않는다 


변동성 : 변동성은 일일 수익률의 표준편차*(연간거래일수)**2 
이때 연간거래일수를 365로 계산 
그러나 연간거래일수는 일반적으로 252일로 계산 

샤프비율 : 연환산 수익률 - 무위험 수익률 /연환산 변동성
보고서의 샤프비율 어떻게 구했는지 확인 필요
무위험 수익률을 무엇으로 정했는지, 샤프비율은 연환산단위인지 누적단위인지 

Winning Ratio : 8186 전일대비 BM수익률과 다른 지수의 전일대비 차가 0인 날짜들은 쉬는 날이라 판단하고 이 날짜들을 필터링해야 하는 과정필요할듯 
거래 수가 아닌 날짜를 기준으로 함
이익 거래일 / 전체 거래일 * 100%
'전일대비수익률', '전일대비BM수익률(%)'칼럼의 경우 소수점 두번째자리까지만 나타나기에 0이 아닌 값을 0으로 나타내는 경우가 많았음
정확한 계산을 위해 데이터에 있는 '전일대비수익률', '전일대비BM수익률(%)'칼럼을 사용하지 않고
직접 일일 수익률을 계산하여 Winning Ratio 를 구했으나 보고서의 숫자와 차이가 많이 나옴.

실제 거래일만을 고려하기 위해 kospi지수에서 이전 kospi 지수와의 차이가 0인 날을 제외함.
수정기준가는 휴일에도 변동가능하기 때문에 kospi지수를 기준으로 함
이경우 winning ratio 가 펀드 53% 코스피 49% 라는 값이 나옴

값이 너무 차이가 심하여 혹시 휴일을 고려하지 않았던 것이 아닐까 하여 휴일을 고려하지 않은 전체기간에 대한 winning ratio를 구해봄
이 경우 winning ratio 가 펀드 60% 코스피 33% 로 나타남


MDD : MAX(최대치-최소치)/최대치
여기서 최대치는 기간중 최대 펀드 가치
최소치는 그 이후의 최소 펀드가치

3. 월간 수익률 monthly_return
펀드 : 8186
KOSPI : 8186
초과수익 : 펀드 - KOSPI
월말일을 기준으로 함 
뭘 초의 값들을 구할 필요 없음
2021년7월의 코스피의 경우 보고서에서는 7월28일의 지수를 기준으로 사용함.
그러나 데이터에서는 7월 28일의 값이 없기에 코드에서는 7월 29일의 지수를 적용

데이터 프레임의 모양은 만들었으나 YTD를 입력하는 과정에서 조정 필요.
YTD를 정의하는 부분 
그 해의 연초의 값이 없을때 YTD를 구하는 것이 맞는지
2023년도 같은 경우 YTD를 구해야 하지 않는지


4. 펀드와 각종 지수에 대해 수익률 그래프 그리기 

진행 사항
기간별 수익률을 펀드의 기간에 따라 데이터프레임이 다르게 적용되도록 조정했습니다
YTD의 경우 펀드의 기간중 연초의 데이터가 없다면 출력하지 않도록 조정했습니다
성능지표는 누적수익률, 연환산 수익률, 변동성, MDD의 값은 데이터의 기간이 하루 차이나는 것을 고려했을때 일치할 것 같습니다.
샤프비율, Winning ratio 의 경우에는 확인이 필요합니다. 
월간수익률은 펀드의 기간에 따라 나타나는 연도를 다르게 설정했고 빈칸은 NaN으로 처리했습니다. 
YTD의 경우 현재는 모든 연도에 대해 나오도록 설정했습니다. 

option = krw, usd 를 추가했고 기본값을 krw로 설정했습니다. 
만약 option = usd 로 설정한다면 펀드의 누적수익률의 경우 환율을 적용한 값으로 사용했습니다. 
또한 출력되는 데이터프레임을 모두 영어로 설정했고 벤치마크를 KOSPI에서 S&P 500 으로 바뀌도록 설정했습니다 
FinanceDataReader 를 이용하여 환율 데이터를 가져왔습니다
open 과 close 의 중간값을 기준환율로 사용했습니다 
수정기준가에 환율을 적용한 방법
수정기준가 / (USD/KRW) = usd 수정기준가  -> 새로운 수정기준가로 업데이트


질문사항 정리
샤프비율 
(연환산 수익률 / 변동성)으로 샤프비율을 정의하여 펀드와 KOSPI에 대해 0.66, -0.79 라는 값이 나왔습니다. (무위험 수익률 = 0)
그러나 라이프 한국기업 ESG향상 일반사모투자신탁 제1호(수수료 차감전, 2021.07.29~2023.10.31)의 보고서의 경우 
샤프비율의 값이 펀드와 KOSPI에 대해 0.53, -1.05 로 나타납니다. 

winning ratio
수정기준가와 KOSPI지수에 대해 일일 차이를 계산하여 값이 양수인 것과 전체 값의 개수를 계산했습니다
(일일 차이가 양수인 날의 수 / 전체 날짜의 수) 를 winnig ratio 라고 정의했습니다. 
그 결과 펀드 60.07 KOSPI 33.25라는 값이 나왔습니다. 
그러나 보고서의 값에는 펀드 71.4, KOSPI 28.6 의 값이 나옵니다. 

추가로 거래일만을 계산한 것은 아닐까 하여 휴일과 같은 거래일을 모두 제외하고 winning ratio를 구해도 보았습니다.
그러나 펀드 53.16, KOSPI 49.55 라는 값이 나왔습니다. 

YTD의 값의 경우 
연초부터 현재까지의 기간을 나타낸다면 데이터 내에 기간이 짧아 연초의 데이터가 없다면 YTD값을 어떻게 처리할지 궁금합니다
현재 저는 연초의 데이터가 없다면 YTD를 출력하지 않도록 설정한 상태인데 YTD는 항상 나오게 해야 할까요

또한 월간 수익률의 YTD의 경우 2021년의 YTD는 구하고 2023년의 YTD는 구하지 않았는데 이유가 궁금합니다.

option = usd 로 설정한다면 수정기준가에 환율을 적용하였고
FinanceDataReader 를 이용하여 환율 데이터를 가져왔습니다
open 과 close 의 중간값을 기준환율로 사용했습니다 
수정기준가에 환율을 적용한 방법은 다음과 같습니다.
수정기준가 / (USD/KRW) = usd 수정기준가  -> 새로운 수정기준가로 업데이트
이 방법에 대해 오류가 없는지 궁금합니다

현재 코드에서는 누적수익률 그래프에서 주식비중을 표현하기 위해 2160의 데이터도 사용했는데 100004, A00001펀드 말고는 2160데이터가 없어서 다른 펀드들의 2160데이터도 받을수 있을까요

추가적으로 코드에서 피드백 내용부탁드립니다 

1. 샤프비율

2. 승률
winning ratio 의 경우 >= 0 으로 하면 펀드 68.57 코스피 66.14
휴일의 경우 일일차이는 전부 0으로 나오기에 코스피가 매우 높은 값으로 나오는 것같다. 반면 수정기준가는 휴일에도 바뀌었음



보고서의 기간은 2021.07.29 ~ 2023.10.31 이지만 
인덱스의 경우 2021.07.28 종가부터 시작

krw usd 누구의 관점인지 확인 -> 환율을 어떻게 적용할지 정하기 
krw 적용시 spx 와 같은 해외지수에 환율 적용하는지
usd 적용시 한국 지수에 환율 적용
결과물 만들어서 보여주고 이렇게 수식을 세워서 이렇게 적용했습니다 괜찮을까요?

여러 인덱스를 유동적으로 적용가능하도록 설정 
MSCI KOREA같은 지표를 벤치마크로 사용가능하도록 


환율을 적용한 방식 
FinanceDataReader 를 이용하여 환율 데이터를 가져왔습니다
open 과 close 의 중간값을 기준환율로 사용했습니다 
option = usd 로 설정한다면 수정기준가와 한국 인덱스에 환율을 적용하였고
환율을 적용한 방법은 다음과 같습니다.
수정기준가 / (USD/KRW) = usd 수정기준가  -> 새로운 수정기준가로 업데이트
외국인이 한국에 투자할 때 달러를 원화로 바꿀경우를 고려한 것이며 다시 원화에서 달러로 환전하는 경우는 고려되지 않음

option krw 로 설정한다면 usd인덱스에 환율을 적욯하였고 
환율을 적용한 방법은 다음과 같습니다
인덱스 * (USD/KRW) = krw -> 원화로 표현된 인덱스로 업데이트
만약 한국인이 외국에 투자했을때 원화를 달러로 바꿀경우를 고려한 것이며 다시 달러에서 원화로 환전하는 경우는 고려되지 않음

성능지표 계산식 정리
누적수익률 : (end_date 수정기준가 - start_date 수정기준가) / start_date 수정기준가
연환산 수익률 : (누적수익률 * 365) / 총 일수
변동성 : 일일 수익률의 표준편차 * root(365) *100
샤프비율 : ((일일 수익률의 평균 - 무위험 수익률) / 일일 수익률의 표준편차) * root(365)
 * 무위험 수익률은 (CD91 의 평균 / 100) /365 
 winning ratio : 펀드와 코스피의 일일 수익률 비교 
 펀드의 winning ratio : (펀드의 일일 수익률 - 코스피의 일일 수익률).round(3) >= 0 인 일수 / 총 일수
 코스피의 winning ratio : (펀드의 일일 수익률 - 코스피의 일일 수익률).round(3) < 0 인 일수 / 총 일수
MDD : 기간 내 모든 날짜에 대해 (현재 가치 - 이후의 최소 가치) / 현재 가치 의 최댓값

* 특이사항
변동성의 값과 샤프비율의 값 root(370)으로 계산한 값이 가장 정확
winning ratio 또한 최근 370을 기준으로 계산한 값이 가장 정확 


인덱스는 각나라의 currency를 따른다

메뉴얼 작성 
기술 문서 작성 
내가 만든 모듈에는 이런 메서드가 있고 예시 적기

100004펀드와 100008 펀드에 대해 모듈을 실행하였고 다른 결과는 확인함
winning ratio 값 100004 100008 펀드의 값을 맞추기 위해 모든 일자에 대하여 펀드와 코스피의 일일수익률을 가지고 펀드 코스피의 차 의 비율, 펀드, 코스피의 양수의 비율, 반올림 경우의 수를 고려해서 계산해보았으나 보고서의 값을 만족시키는 같은 일자를 찾지 못하였음 
이제는 보고서의 값이 잘못된 것이 아닐까 의심하는 지경에 일으렀습니다 



질문 
100004 펀드와 100008펀드에 대해 모듈을 실행하였고 다른 결과는 다 확인하였는데 
winning ratio의 값이 
저는 펀드 : 71.7, KOSPI : 28.3 값이 나오고 
보고서에는 펀드 : 66.7, KOSPI : 33.3 값으로 표기되어 있습니다 
때문에 100004와 100008에 대해 동시에 만족하는 값을 찾기 위해 모든 일자에 대하여 펀드와 코스피의 일일 수익률을 가지고 
펀드 코스피의 차 의 비율, 펀드, 코스피의 양수의 비율, 반올림 경우의 수를 고려해서 계산해보았으나 보고서의 값을 만족시키는 같은 일자를 찾지 못하였습니다. 
펀드의 winning ratio = (펀드와 코스피의 일일 수익률의 차이가 0이상인 날의 수) / 전체 일수
제가 생각한 수식은 이것이고 다른 수식들도 적용해보았는데 동시에 일치하는 수식을 찾을수 없어서 질문을 올립니다




0


각 종목들의 월 말 정보를 담고 있는 2305파일을 사용하여
각 종목들의 월말 평가액을 스택차트로 표현 





