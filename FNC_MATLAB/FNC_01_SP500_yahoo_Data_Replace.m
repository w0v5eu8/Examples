clearvars;

%% SP500 Data Collection
% 해외 주식 데이터 collection
% 인덱스: S&P 500 (^GSPC), NASDAQ Composite (^IXIC), Dow Jones Industrial Average (^DJI)
% 개별: S&P 500에 속하는 종목
% 종목리스트: https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies
% 데이터: https://finance.yahoo.com/

%% 데이터 파일 체크 및 초기화
% 데이터 파일 mat 이 있나 체크하고 없는 경우 처음 크롤링 하는 것이므로 ETF 변수를 setting
if exist('../../DATA/FNC/SP500.mat','file')
    load('../../DATA/FNC/SP500.mat');
    for i=1:length(SP500)
        SP500(i).Update=0;
    end
else
    SP500.codeName=[];
    SP500.Update=[];
    SP500.Date=[];
    SP500.Price=[];
    SP500.Volume=[];
    SP500.Close=[];
    SP500.Open=[];
    SP500.High=[];
    SP500.Low=[];
end

%% Symbol 리스트 설정
options = weboptions('Timeout',Inf);
flag=1;
while flag==1
    try
        str=webread('https://en.m.wikipedia.org/wiki/List_of_S&P_500_companies',options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end

% 위키 페이지 파싱
str_all=[];
str_all=[str_all strfind(str,'XNYS:')];
str_all=[str_all strfind(str,'mbol/')];
str_all=[str_all strfind(str,'bols/')];

SP500_list(1).Symbol='^GSPC'; % S&P 500
SP500_list(2).Symbol='^IXIC'; % NASDAQ
SP500_list(3).Symbol='^DJI'; % DOW JONES

% 위키 페이지 파싱
for i=1:length(str_all)
    temp_str=sprintf('%s',str(str_all(i)+5:str_all(i)+15));
    str_i=strfind(temp_str,'"');
    complete_str=sprintf('%s',temp_str(1:str_i-1));
    complete_str=upper(complete_str);
    complete_str(strfind(complete_str,'.'))='-';
    SP500_list(3+i).Symbol=complete_str;
end
fprintf('SP500 list done.\n');

%% World_List 순서대로 업데이트
for i=1:length(SP500_list)
    
    %% 크롤링 전 setting
    % SP500_list에 있는 주식이 SP500의 몇번째 index인지 찾기.
    % SP500에 있는 경우  stock_exist는 1, stock_index는 SP500의 index 번호
    % SP500없는 신규 상장된 주식이면 stock_exist는 0, stock_index는 SP500의 length+1
    stock_exist=0;
    for stock_index=1:length(SP500)
        if(strcmp(SP500_list(i).Symbol,SP500(stock_index).codeName))
            stock_exist=1;
            break;
        end
    end
    if (stock_exist == 0)
        if isempty(SP500(1).codeName)
            stock_index = 1;
        else
            stock_index=length(SP500)+1;
        end
    end
    
    % 현재 상장된 종목이면 Update를 1로 설정
    SP500(stock_index).Update=1;
    
    % yahoo 크롤링 코드는 시작일만 지정하면 최신데이터까지 크롤링 됨.
    % FNC_01_SP500_yahoo_Data_Replace.m은 % 수정 주가 반영을 위해서 전체 데이터를 새로 받아 옮.
    % 따라서 fromdate는 19000101로 설정,
    % SP500에 날짜 데이터는 double형 yyyymmdd 형식으로 저장되어 있음
    % 데이타가 없는 기간도 있음. empty인지 체크 필요.
    
    fromdate='01011900';
    todate=datestr(now,'ddmmyyyy');
    % 불러올 SP500 심볼 설정
    symbol= SP500_list(i).Symbol;
    
    %% SP500 개별 크롤링
    % fromdate가 todate (now) 보다 뒷날이면 pass
    if datenum(fromdate,'ddmmyyyy') <= datenum(todate,'ddmmyyyy')
        options = weboptions('Timeout',Inf);
        flag=1;
        while flag==1
            try
                Temp=FNC_Func_yahoo(fromdate, todate, symbol);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
    else
        fprintf('Warninig: SP500 %d / %d - %s fromdate Pass.\n',i, length(SP500_list), SP500_list(i).Symbol);
    end
    
    %% 크롤링 데이터 SP500에 업데이트
    SP500(stock_index).codeName=SP500_list(i).Symbol;
    
    if datenum(fromdate,'ddmmyyyy') <= now % fromdate가 todate 보다 뒷날이면 pass
        if isempty(Temp) == 0 % 크롤링된 데이터가 없으면 pass
            SP500(stock_index).Date=[str2num(datestr(datenum(cellstr(Temp.Date),'yyyy-mm-dd'),'yyyymmdd'))]; % 년월일
            SP500(stock_index).Close=[Temp.Close]; % 종가
            SP500(stock_index).Open=[Temp.Open]; % 시가
            SP500(stock_index).High=[Temp.High]; % 고가
            SP500(stock_index).Low=[Temp.Low]; % 저가
            SP500(stock_index).Volume=[Temp.Volume]; % 거래량
            SP500(stock_index).Price=[Temp.AdjClose]; % 수정 종가
        else
            fprintf('Warninig: SP500 %d / %d - %s Data empty.\n',i, length(SP500_list), SP500_list(i).Symbol);
        end
    end
    
    %% NaN Check
    if sum(isnan(SP500(stock_index).Price)) > 0
        fprintf('Warninig: SP500 %d / %d - %s NaN\n',i, length(SP500_list), SP500_list(i).Symbol);
        for nan_i=1:length(SP500(stock_index).Price)
            if isnan(SP500(stock_index).Price(nan_i)) == 1
                fprintf('Warninig: SP500 %d / %d - %s NaN Adjust index = %d\n',i, length(SP500_list), SP500_list(i).Symbol, nan_i);
                if nan_i == 1
                    SP500(stock_index).Price(nan_i) = 0;
                else
                    SP500(stock_index).Price(nan_i) = SP500(stock_index).Price(nan_i-1);
                end
            end
        end
    end
    
    fprintf('SP500: %d / %d - %s Done.\n',i, length(SP500_list), SP500_list(i).Symbol);
end

%% Save
save('../../DATA/FNC/SP500.mat','SP500');
fprintf('FNC_01_SP500 Done.\n');