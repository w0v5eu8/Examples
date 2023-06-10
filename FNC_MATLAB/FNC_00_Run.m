clc;
tic
filename=sprintf('../../DATA/FNC/Log_%s.txt',datestr(now-1,'yyyymmdd'));
diary(filename);
diary on;

%% KRX Data Collection
% 한국 주식시장 시간 09:00 ~ 15:30
% krx 사이트에서 개별 종목 일자별 데이터는 자정이후에 오늘 데이터가 업데이트 됨
% krx 사이트에서 크롤링 차단을 위해 접속 IP가 차단되는 경우가 있음
% 이럴때는 무료 proxy 서버를 설정해서 접속해야 함.
% Matlab->홈->기본설정->'MATLAB'탭->'웹'탭->인터넷연결에서 설정
% 무료 프록시 서버 리스트 HTTP Protocol로 연결해야 함.
% http://www.freeproxylists.net/
FNC_01_KRX_Index_Data % 자정 이후 KRX 사이트가 업데이트 됨
FNC_01_KOSPI_KOSDAQ_Data % 자정 이후 KRX 사이트가 업데이트 됨
FNC_01_ETF_Data % 실시간으로 KRX 사이트가 업데이트 됨

%% KRX Preprocessing: Scaling, PST 
FNC_02_Preprocessing('KRX_Index')
FNC_02_Preprocessing('KRX')
FNC_02_Preprocessing('ETF')

%% Done
toc_time=toc;
fprintf('toc time = %f min\n',toc_time/60);
diary off;