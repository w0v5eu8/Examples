clearvars;

%% KRX(KOSPI KOSDAQ) Data Collection
% 주식 전체리스트 와 개별 PER/PBR 등 : http://marketdata.krx.co.kr/mdi#document=13020401
% 주식 일자별 시세: http://marketdata.krx.co.kr/mdi#document=040204
% 주식 일자별 시세 페이지는 당일날 종목의 종가 시가 고가 저가는 보조 테이블로 보여주고
% 본 테이블은 전날까지의 데이터만 있음. 또한 보조 테이블은 시가총액 등은 안나옴.
% 따라서 자정 12시가 지난 후에 크롤링 해야 당일 데이터를 정확하게 받을 수 있음.

%% 데이터 파일 체크 및 초기화
% 데이터 파일 mat 이 있나 체크하고 없는 경우 처음 크롤링 하는 것이므로 KRX 변수를 setting
if exist('../../DATA/FNC/KRX.mat','file')
    load('../../DATA/FNC/KRX.mat');
    for i=1:length(KRX)
        KRX(i).Update=0;
    end
else
    KRX.full_code=[];
    KRX.shrot_code=[];
    KRX.codeName=[];
    KRX.marketName=[];
    KRX.Update=[];
    KRX.Date1=[];
    KRX.manageFlag=[];
    KRX.EPS=[];
    KRX.PER=[];
    KRX.BPS=[];
    KRX.PBR=[];
    KRX.DPS=[];
    KRX.DYR=[];
    KRX.Date=[];
    KRX.Close=[];
    KRX.Volume=[];
    KRX.vPrice=[];
    KRX.Open=[];
    KRX.High=[];
    KRX.Low=[];
    KRX.Price=[]; % 시가 총액
    KRX.fShare=[];
end

%% 현재 KOSPI, KOSDAQ 상장 종목 종목코드 불러오기
% krx 홈페이지 코드 분석 캡쳐 참고 (크롬에서 F12 눌러서 Network 패킷 분석)
options = weboptions('Timeout',Inf);
flag=1;
while flag==1
    try
        krx_code=webread('http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx', ...
            'bld','COM/finder_stkisu', ...
            'name','form', ...
            options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end
options = weboptions('RequestMethod','post','Timeout',Inf, ...
    'HeaderFields', {'Referer' 'http://marketdata.krx.co.kr/mdi'; ...
    'User-Agent' '"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome'});
flag=1;
while flag==1
    try
        temp=webread('http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx', ...
            'pagePath','/contents/COM/FinderStkIsu.jsp',...
            'mktsel','ALL', ...
            'no','P1', ...
            'code',krx_code,...
            'pageFirstCall', 'Y', ...
            options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end
temp=jsondecode(temp);
temp_list=temp.block1;

fprintf('KOSPI KOSDAQ list done.\n');

%% 상장된 종목(temp_list)들을 KRX에 순서대로 업데이트
for i=1:length(temp_list)
    
    %% 크롤링 전 setting
    % temp_list에 있는 주식이 KRX의 몇번째 index인지 찾기.
    % KRX에 있는 경우  stock_exist는 1, stock_index는 KRX의 index 번호
    % KRX없는 신규 상장된 주식이면 stock_exist는 0, stock_index는 KRX의 length + 1
    stock_exist=0;
    for stock_index=1:length(KRX)
        if(strcmp(temp_list(i).full_code,KRX(stock_index).full_code))
            stock_exist=1;
            break;
        end
    end
    if (stock_exist == 0)
        if isempty(KRX(1).codeName)
            stock_index = 1;
        else
            stock_index=length(KRX)+1;
        end
    end
    
    % 현재 상장된 종목이면 Update를 1로 설정
    KRX(stock_index).Update=1;
    
    % fromdate는  존재하는 데이터의  마지막 날+1, 신규 상장된 종목이면 19700101로 설정,
    % todate는 크롤링 기준 일자(오늘)
    % KRX에 날짜 데이터는 double형 yyyymmdd 형식으로 저장되어 있음
    % krx의 (개별 PER/PBR 등 페이지)와 (주식 일자별 시세 페이지)의 같은 기간 데이터 수가 다름.
    % 따라서 date1, date 두개를 운영.
    % 데이타가 없는 기간도 있음. empty인지 체크 필요.
    if (stock_exist)
        if isempty(KRX(stock_index).Date1) == 0
            fromdate1=datestr(datenum(num2str(KRX(stock_index).Date1(length(KRX(stock_index).Date1))),...
                'yyyymmdd')+1,'yyyymmdd');
        end
        if isempty(KRX(stock_index).Date) == 0
            fromdate=datestr(datenum(num2str(KRX(stock_index).Date(length(KRX(stock_index).Date))),...
                'yyyymmdd')+1,'yyyymmdd');
        end
    else
        fromdate1='19700101';
        fromdate='19700101';
    end
    todate=datestr(now-1,'yyyymmdd');
    
    % 불러올 개별 종목 id 설정
    isu_cd = temp_list(i).full_code;
    isu_srt_cd = temp_list(i).short_code;
    
    %% 개별 PER/PBR 등 크롤링: http://marketdata.krx.co.kr/mdi#document=13020401
    if datenum(fromdate1,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        options = weboptions('Timeout',Inf);
        flag=1;
        while flag==1
            try
                krx_code=webread('http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx', ...
                    'bld','MKD/13/1302/13020401/mkd13020401', ...
                    'name','chart', ...
                    options);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
        options = weboptions('RequestMethod','post','Timeout',Inf, ...
            'HeaderFields', {'Referer' 'http://marketdata.krx.co.kr/mdi'});
        flag=1;
        while flag==1
            try
                temp=webread('http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx', ...
                    'market_gubun','ALL', ...
                    'isu_cd',isu_cd, ...
                    'isu_srt_cd',isu_srt_cd, ...
                    'fromdate',fromdate1, ...
                    'todate',todate, ...
                    'pagePath','/contents/MKD/13/1302/13020401/MKD13020401.jsp', ...
                    'code',krx_code, ...
                    options);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
        temp=jsondecode(temp);
        temp_data1=temp.result;
    else
        fprintf('Warning: KRX %d / %d - %s fromdate1 Pass.\n',i, length(temp_list), temp_list(i).codeName);
    end
    
    %% 주식 일자별 시세 크롤링: http://marketdata.krx.co.kr/mdi#document=040204
    if datenum(fromdate,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        options = weboptions('Timeout',Inf);
        flag=1;
        while flag==1
            try
                krx_code=webread('http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx', ...
                    'bld','MKD/13/1302/13020103/mkd13020103_02', ...
                    'name','chart', ...
                    options);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
        options = weboptions('RequestMethod','post','Timeout',Inf, ...
            'HeaderFields', {'Referer' 'http://marketdata.krx.co.kr/mdi'});
        flag=1;
        while flag==1
            try
                temp=webread('http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx', ...
                    'isu_cd',isu_cd, ...
                    'isu_srt_cd',isu_srt_cd, ...
                    'fromdate',fromdate, ...
                    'todate',todate, ...
                    'pagePath','/contents/MKD/04/0402/04020100/MKD04020100T3T2.jsp', ...
                    'code',krx_code, ...
                    options);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
        temp=jsondecode(temp);
        temp_data2=temp.block1;
    else
        fprintf('Warning: KRX  %d / %d - %s fromdate Pass.\n',i, length(temp_list), temp_list(i).codeName);
    end
    
    %% 크롤링 데이터 KRX에 업데이트
    KRX(stock_index).full_code=temp_list(i).full_code; % 종목 full 코드
    KRX(stock_index).shrot_code=temp_list(i).short_code; % 종목 short 코드
    KRX(stock_index).codeName=temp_list(i).codeName; % 종목명
    KRX(stock_index).marketName=temp_list(i).marketName; % KOSPI or KOSDAQ
    
    if datenum(fromdate1,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        if isempty(temp_data1) == 0 % 크롤링된 데이터가 없으면 pass
            KRX(stock_index).Date1=[KRX(stock_index).Date1; ...
                str2num(datestr(datenum(flip({temp_data1.work_dt}.'),'yyyy/mm/dd'),'yyyymmdd'))]; % 년월일
            KRX(stock_index).manageFlag=[KRX(stock_index).manageFlag; ...
                strcmp(flip({temp_data1.iisu_code}.'),'-')]; % 관리여부
            KRX(stock_index).EPS=[KRX(stock_index).EPS; ...
                str2double(flip({temp_data1.prv_eps}.'))]; % EPS
            KRX(stock_index).PER=[KRX(stock_index).PER; ...
                str2double(flip({temp_data1.per}.'))]; % PER
            KRX(stock_index).BPS=[KRX(stock_index).BPS; ...
                str2double(flip({temp_data1.bps}.'))]; % BPS
            KRX(stock_index).PBR=[KRX(stock_index).PBR; ...
                str2double(flip({temp_data1.pbr}.'))]; % PBR
            KRX(stock_index).DPS=[KRX(stock_index).DPS; ...
                str2double(flip({temp_data1.stk_dvd}.'))]; % 주당배당금
            KRX(stock_index).DYR=[KRX(stock_index).DYR; ...
                str2double(flip({temp_data1.dvd_yld}.'))]; % 배당수익률
        else
            fprintf('Warning: KRX  %d / %d - %s Data1 empty.\n',i, length(temp_list), temp_list(i).codeName);
        end
    end
    if datenum(fromdate,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        if isempty(temp_data2) == 0 % 크롤링된 데이터가 없으면 pass
            KRX(stock_index).Date=[KRX(stock_index).Date; ...
                str2num(datestr(datenum(flip({temp_data2.trd_dd}.'),'yyyy/mm/dd'),'yyyymmdd'))]; % 년월일
            KRX(stock_index).Close=[KRX(stock_index).Close; ...
                str2double(flip({temp_data2.tdd_clsprc}.'))]; % 종가
            KRX(stock_index).Volume=[KRX(stock_index).Volume; ...
                str2double(flip({temp_data2.acc_trdvol}.'))]; % 거래량
            KRX(stock_index).vPrice=[KRX(stock_index).vPrice; ...
                str2double(flip({temp_data2.acc_trdval}.'))]; % 거래대금
            KRX(stock_index).Open=[KRX(stock_index).Open; ...
                str2double(flip({temp_data2.tdd_opnprc}.'))]; % 시가
            KRX(stock_index).High=[KRX(stock_index).High; ...
                str2double(flip({temp_data2.tdd_hgprc}.'))]; % 고가
            KRX(stock_index).Low=[KRX(stock_index).Low; ...
                str2double(flip({temp_data2.tdd_lwprc}.'))]; % 저가
            KRX(stock_index).Price=[KRX(stock_index).Price; ...
                str2double(flip({temp_data2.mktcap}.'))]; % 시가총액(백만)
            KRX(stock_index).fShare=[KRX(stock_index).fShare; ...
                str2double(flip({temp_data2.list_shrs}.'))]; % 상장주식수
        else
            fprintf('Warning: KRX  %d / %d - %s Data2 empty.\n',i, length(temp_list), temp_list(i).codeName);
        end
    end
    
    %% NaN Check
    if sum(isnan(KRX(stock_index).Price)) > 0
        fprintf('Warning: KRX  %d / %d - %s NaN\n',i, length(temp_list), temp_list(i).codeName);
        for nan_i=1:length(KRX(stock_index).Price)
            if isnan(KRX(stock_index).Price(nan_i)) == 1
                fprintf('Warning: KRX %d / %d - %s NaN Adjust index = %d\n',i, length(temp_list), temp_list(i).codeName, nan_i);
                if nan_i == 1
                    KRX(stock_index).Price(nan_i) = 0;
                else
                    KRX(stock_index).Price(nan_i) = KRX(stock_index).Price(nan_i-1);
                end
            end
        end
    end
    
    fprintf('KRX: %d / %d - %s Done.\n',i, length(temp_list), temp_list(i).codeName);
end

%% Save
save('../../DATA/FNC/KRX.mat','KRX');
fprintf('FNC_01_KOSPI_KOSDAQ_Data Done.\n');