clearvars;

%% ETF Data Collection
% ETF 전체리스트: http://marketdata.krx.co.kr/mdi#document=13040101
% ETF 종목별: http://marketdata.krx.co.kr/mdi#document=08010203

%% 데이터 파일 체크 및 초기화
% 데이터 파일 mat 이 있나 체크하고 없는 경우 처음 크롤링 하는 것이므로 ETF 변수를 setting
if exist('../../DATA/FNC/ETF.mat','file')
    load('../../DATA/FNC/ETF.mat');
    for i=1:length(ETF)
        ETF(i).Update=0;
    end
else
    ETF.full_code=[];
    ETF.shrot_code=[];
    ETF.codeName=[];
    ETF.Update=[];
    ETF.Date=[];
    ETF.Price=[]; % 종가
    ETF.NAV=[];
    ETF.Basic=[];
    ETF.Volume=[];
    ETF.vPrice=[];
    ETF.Open=[];
    ETF.High=[];
    ETF.Low=[];
end

%% 현재 ETF 상장 종목 종목코드 불러오기
options = weboptions('Timeout',Inf);
flag=1;
while flag==1
    try
        etf_code=webread('http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx', ...
            'bld','MKD/13/1304/13040101/mkd13040101', ...
            'name','form', ...
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
            'pagePath','/contents/MKD/13/1304/13040101/MKD13040101.jsp',...
            'domforn','00', ...
            'uly_gubun','00', ...
            'gubun','00',...
            'fromdate',datestr(now-30,'yyyymmdd'), ...
            'todate',datestr(now,'yyyymmdd'), ...
            'isu_cd','ALL', ...
            'code',etf_code,...
            options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end
temp=jsondecode(temp);
temp_list=temp.block1;

fprintf('ETF list done.\n');

%% 상장된 종목(temp_list)들을 ETF에 순서대로 업데이트
for i=1:length(temp_list)
    
    %% 크롤링 전 setting
    % temp_list에 있는 주식이 ETF의 몇번째 index인지 찾기.
    % ETF에 있는 경우  stock_exist는 1, stock_index는 ETF의 index 번호
    % ETF없는 신규 상장된 주식이면 stock_exist는 0, stock_index는 ETF의 length+1
    stock_exist=0;
    for stock_index=1:length(ETF)
        if(strcmp(temp_list(i).isu_kor_abbrv,ETF(stock_index).codeName))
            stock_exist=1;
            break;
        end
    end
    if (stock_exist == 0)
        if isempty(ETF(1).codeName)
            stock_index = 1;
        else
            stock_index=length(ETF)+1;
        end
    end
    
    % 현재 상장된 종목이면 Update를 1로 설정
    ETF(stock_index).Update=1;
    
    % fromdate는  존재하는 데이터의  마지막 날+1, 신규 상장된 종목이면 19700101로 설정,
    % todate는 크롤링 기준 일자(오늘)
    % ETF에 날짜 데이터는 double형 yyyymmdd 형식으로 저장되어 있음
    % 데이타가 없는 기간도 있음. empty인지 체크 필요.
    if (stock_exist)
        if isempty(ETF(stock_index).Date) == 0
            fromdate=datestr(datenum(num2str(ETF(stock_index).Date(length(ETF(stock_index).Date))),...
                'yyyymmdd')+1,'yyyymmdd');
        end
    else
        fromdate='19700101';
    end
    todate=datestr(now-1,'yyyymmdd');
    
    % 불러올 ETF id 설정
    isu_cd = temp_list(i).isu_cd;
    
    %% ETF 개별 크롤링: http://marketdata.krx.co.kr/mdi#document=08010203
    if datenum(fromdate,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        options = weboptions('Timeout',Inf);
        flag=1;
        while flag==1
            try
                etf_code=webread('http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx', ...
                    'bld','MKD/08/0801/08010700/mkd08010700_04', ...
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
                    'mkt','ETF', ...
                    'domforn','00', ...
                    'uly_gubun','00', ...
                    'gubun','00', ...
                    'isu_cd',isu_cd, ...
                    'isu_srt_cd',isu_cd, ...
                    'fromdate',fromdate, ...
                    'todate',todate, ...
                    'gubun2','2', ...
                    'pagePath','/contents/MKD/08/0801/08010700/MKD08010700.jsp', ...
                    'code',etf_code, ...
                    options);
                flag=0;
            catch
                fprintf('re try webread\n');
                flag=1;
            end
        end
        temp=jsondecode(temp);
        temp_data=temp.block1;
    else
        fprintf('Warninig: ETF %d / %d - %s fromdate Pass.\n',i, length(temp_list), temp_list(i).isu_kor_abbrv);
    end
    
    %% 크롤링 데이터 ETF에 업데이트
    ETF(stock_index).full_code=temp_list(i).isu_cd; % 종목 full 코드
    ETF(stock_index).shrot_code=temp_list(i).isu_srt_cd; % 종목 short 코드
    ETF(stock_index).codeName=temp_list(i).isu_kor_abbrv; % 종목명
    
    if datenum(fromdate,'yyyymmdd') <= datenum(todate,'yyyymmdd') % fromdate가 todate 보다 뒷날이면 pass
        if isempty(temp_data) == 0 % 크롤링된 데이터가 없으면 pass
            ETF(stock_index).Date=[ETF(stock_index).Date; ...
                str2num(datestr(datenum(flip({temp_data.work_dt}.'),'yyyy/mm/dd'),'yyyymmdd'))]; % 년월일
            ETF(stock_index).Price=[ETF(stock_index).Price; ...
                str2double(flip({temp_data.isu_end_pr}.'))]; % 종가
            ETF(stock_index).NAV=[ETF(stock_index).NAV; ...
                str2double(flip({temp_data.last_nav}.'))]; % NAV
            ETF(stock_index).Basic=[ETF(stock_index).Basic; ...
                str2double(flip({temp_data.last_indx}.'))]; % 기초지수
            ETF(stock_index).Volume=[ETF(stock_index).Volume; ...
                str2double(flip({temp_data.tot_tr_vl}.'))]; % 거래량(주)
            ETF(stock_index).vPrice=[ETF(stock_index).vPrice; ...
                str2double(flip({temp_data.tot_tr_amt}.'))]; % 거래대금(백만원)
            ETF(stock_index).Open=[ETF(stock_index).Open; ...
                str2double(flip({temp_data.isu_opn_pr}.'))]; % 시가
            ETF(stock_index).High=[ETF(stock_index).High; ...
                str2double(flip({temp_data.isu_hg_pr}.'))]; % 고가
            ETF(stock_index).Low=[ETF(stock_index).Low; ...
                str2double(flip({temp_data.isu_lw_pr}.'))]; % 저가
        else
            fprintf('Warninig: ETF %d / %d - %s Data empty.\n',i, length(temp_list), temp_list(i).isu_kor_abbrv);
        end
    end
    
    %% NaN Check
    if sum(isnan(ETF(stock_index).Price)) > 0
        fprintf('Warninig: ETF %d / %d - %s NaN\n',i, length(temp_list), temp_list(i).isu_kor_abbrv);
        for nan_i=1:length(ETF(stock_index).Price)
            if isnan(ETF(stock_index).Price(nan_i)) == 1
                fprintf('Warninig: ETF %d / %d - %s NaN Adjust index = %d\n',i, length(temp_list), temp_list(i).isu_kor_abbrv, nan_i);
                if nan_i == 1
                    ETF(stock_index).Price(nan_i) = 0;
                else
                    ETF(stock_index).Price(nan_i) = ETF(stock_index).Price(nan_i-1);
                end
            end
        end
    end
    
    fprintf('ETF: %d / %d - %s Done.\n',i, length(temp_list), temp_list(i).isu_kor_abbrv);
end

%% Save
save('../../DATA/FNC/ETF.mat','ETF');
fprintf('FNC_01_ETF Done.\n');