clc;
tic
filename=sprintf('../../DATA/FNC/Log_E_%s.txt',datestr(now-1,'yyyymmdd'));
diary(filename);
diary on;

%% 해외주식 Data Collection 미국 주식시장 시간 (한국시간) 23:30 ~ 06:00
% new date 데이터만 업데이트
% FNC_01_SP500_yahoo_Data

% 수정 주가 반영을 위해서 전체 데이터를 새로 받아 옮.
FNC_01_SP500_yahoo_Data_Replace 

%% Preprocessing: Scaling, PST
FNC_02_Preprocessing('SP500')

%% Done
toc_time=toc;
fprintf('toc time = %f min\n',toc_time/60);
diary off;

%% exmple
% struct find
% find([SP500.Update]==0)