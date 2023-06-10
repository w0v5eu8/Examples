# K-project_2021-2

> 파일 설명

+ 📁 data_collection
  + KRX_fullcode_1.ipynb : KRX 상장된 주식 전종목 fullcode 수집 (초기 1회 실행)
  + KRX_fullcode_2.ipynb : KRX 주식 전종목 상태 (상장폐지, 신규상장 등) 매일 업데이트
  + datacollection_1.ipynb : KRX의 주식 전종목 날짜별 데이터 수집 (초기에 1회 실행, 약 6시간 소요)
  + datacollection_2.ipynb : KRX의 주식 전종목 날짜별 데이터 매일 업데이트 (하루에 1번 실행, 약 1시간 20분 소요)
 
+ 📁 data_preprocessing
  + AWS_preprocessing.ipynb : AWS scikit-learn 이용한 주가데이터 병렬처리
  + preprocessing.py : AWS Scikit-Learn 전처리 스크립트
  + TimeSeries_trend.ipynb : 시가총액 trend [-1,1] sell-buy signal 값으로 변환
  + optimization.ipynb : 보조지표 파라미터 최적화 및 sell/buy signal 도출

+ 📁 modeling
  + ensemble.py : AWS Scikit-Learn 전처리 스크립트 (Elasticnet, RF, XGBoost)
  + final_model_module.ipynb : LSTM, GRU, CNN+GRU 모델
