function FNC_02_Preprocessing(Type)

%% 데이터 로딩
cmd=sprintf('load(''../../DATA/FNC/%s.mat''); DATA=%s;', Type, Type);
eval(cmd)

%% KOSPI KOSDAQ는 시가총액, 나머지는 종가를 기준으로 가격 값을 일정 범위로 스케일링
% KOSPI KOSDAQ 경우 증자 감자가 때문에 가격이 급격하게 변할 수 있으므로 시가총액을 기준으로 함.
% A1 -> z-score with mean 0 and standard deviation 1
% A2 -> [0 1] : 이게 PST가 가장 잘나옴.
% A3 -> [-1 1]

%% KRX_Index Scaling
for i=1:length(DATA)
    DATA(i).A1=normalize(DATA(i).Price);
    DATA(i).A2=normalize(DATA(i).Price,'range',[0 1]);
    DATA(i).A3=normalize(DATA(i).Price,'range',[-1 1]);
    fprintf('%s Scaling: %d / %d - %s Done.\n', Type, i, length(DATA), DATA(i).codeName);
end

%% 주가의 Target으로 사용할 Price_Series_Transform
% 지그재그형 주가를 T, P를 기준으로 smooting

%% PST 파라매터 설정
T=5; % time interval
P=0.05; % percentage

%% PST
for j=1:length(DATA)
    % nPST=0 means there is no BUY and SELL point by PST
    if isempty(DATA(j).A2) || length(DATA(j).A2) <= 10
        % too short time series to calculate pst
        DATA(j).PST=[];
        DATA(j).nPST=0;
    else
        [DATA(j).PST, DATA(j).nPST] = FNC_Func_PST(T,P,DATA(j).A2);
    end
    
    fprintf('%s PST: %d / %d - %s Done.\n', Type, j, length(DATA), DATA(j).codeName);
end

%% Save
cmd=sprintf('%s=DATA; save(''../../DATA/FNC/%s_%s.mat'',''%s''); save(''../../DATA/FNC/%s.mat'',''%s'');', ...
    Type, Type, num2str(DATA(1).Date(length(DATA(1).Date))), Type, Type, Type);
eval(cmd)
fprintf('FNC_02_Preprocessing Done.\n');