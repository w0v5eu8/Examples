clearvars;

%% CrytCoin Data Collection
% Bithumb
% https://m.bithumb.com/trade/chart/BTC
% datetime(1388070000.000, 'ConvertFrom', 'posixtime')
% datetime(1388070000.000, 'ConvertFrom', 'posixtime','Timezone','Asia/Seoul')
% datestr(datetime(1388070000.000, 'ConvertFrom', 'posixtime','Timezone','Asia/Seoul'),'yyyymmdd')

% Poloniex
% https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1262340239&end=1572775453&period=86400
% https://docs.poloniex.com/#returntradehistory-public
% https://docs.poloniex.com/#currency-pair-ids

%% 데이터 파일 체크 및 초기화
% 데이터 파일 mat 이 있나 체크하고 없는 경우 처음 크롤링 하는 것이므로 ETF 변수를 setting
if exist('../../DATA/FNC/CC.mat','file')
    load('../../DATA/FNC/CC.mat');
    for i=1:length(CC)
        CC(i).Update=0;
    end
else
    CC.codeName=[];
    CC.full_code=[];
    CC.Update=[];
    CC.Date=[];
    CC.Price=[]; % 종가
    CC.Open=[];
    CC.High=[];
    CC.Low=[];
    CC.Volume=[];
    
    % CrytCoin은 아래 종목만 데이터 수집. 시가총액 상위종목.
    CC(1).codeName='비트코인';
    CC(1).full_code='BTC';
    CC(2).codeName='이더리움';
    CC(2).full_code='ETH';
    CC(3).codeName='리플';
    CC(3).full_code='XRP';
    CC(4).codeName='비트코인 캐시';
    CC(4).full_code='BCH';
    CC(5).codeName='라이트코인';
    CC(5).full_code='LTC';
    CC(6).codeName='이오스';
    CC(6).full_code='EOS';
    CC(7).codeName='비트코인에스브이';
    CC(7).full_code='BSV';
    CC(8).codeName='스텔라루멘';
    CC(8).full_code='XLM';
    CC(9).codeName='트론';
    CC(9).full_code='TRX';
    CC(10).codeName='에이다';
    CC(10).full_code='ADA';
    CC(11).codeName='체인링크';
    CC(11).full_code='LINK';
    CC(12).codeName='이더리움 클래식';
    CC(12).full_code='ETC';
    CC(13).codeName='퀀텀';
    CC(13).full_code='QTUM';
    CC(14).codeName='제로엑스';
    CC(14).full_code='ZRX';
    CC(15).codeName='어거';
    CC(15).full_code='REP';
    for i=1:length(CC)
        CC(i).Update=0;
    end
end

%%
options = weboptions('RequestMethod','get','Timeout',Inf);
flag=1;
while flag==1
    try
        b_code=webread('https://c.go-mpulse.net/api/config.json?key=9DRDT-FACN7-E8CC2-ZHTLM-GTPUK&d=m.bithumb.com&t=5243425&v=1.571.0&if=&sl=1&si=040c7f51-435b-47ed-8bcc-1457915203a1-q0jesn&r=&bcn=%2F%2F60062f0a.akstat.io%2F&acao=', ...
            options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end

options = weboptions('RequestMethod','post','Timeout',Inf, ...
    'HeaderFields', {'origin' 'https://m.bithumb.com'; ...
    'cookie' 'MONID=gjHnbZSPMkk; csrf_xcoin_name=0a27da1a3f215e7f634f6e5682738c91; lang=korean; _ga=GA1.2.1205221154.1573026216; _gid=GA1.2.1588526323.1573026216; _fbp=fb.1.1573026216592.618355928; AF_BANNERS_SESSION_ID=1573026218590; ak_bmsc=C93763DF11BB304EED431C184FF5D371ADDFE35633700000A879C25DE70B290A~plCdKgDuuvOKW/kA/SF+Q7KrHvOVCljirPsdbuUqIv+EGJ5u93N7RJJWwjW4w0I1sBz6E7fnnWFoIBgMRtZAmUR3eGyitW4N3Ylxw2KfvNv0aMKTmkQIiDAN3/v8vQIAGC8zH5hhgDi4XMExEriP10KyxFaJJRuzVzEtIH3RK+atV1ggWaaHxL68vnUx2SWOiNA0FG21+A+dEyDFy3xpW9hrx0zHamkzf/vyjxoAPZ1l0D0afpuuJfy8RSDxUyBjGpsKogM7wT+MiAxOfah6VZxg==; wcs_bt=s_5a2dea14ad2b:1573026814; RT="sl=5&ss=1573026214738&tt=5799&obo=0&bcn=%2F%2F60062f0a.akstat.io%2F&sh=1573027633731%3D5%3A0%3A5799%2C1573027620027%3D4%3A0%3A5786%2C1573026815664%3D3%3A0%3A5677%2C1573026814945%3D2%3A0%3A4173%2C1573026218599%3D1%3A0%3A3853&dm=bithumb.com&si=040c7f51-435b-47ed-8bcc-1457915203a1"; xcoin_session=f11e2a54095959fd8f00e47d6d872882ac97a886f52834f19257f500ee539f76547bd69b431b26cfb625fa5fc0a5490561c69feed895dcaddcaf403a362b10bacYErLOzOWNGSc%2FJJNw2Gw%2F6gfgmFN%2F2j3myDXj6GuiqDZ0CIyS4gkGf7bLj9b3wbeD2P0yGyFxVAuW1Vb%2BiFfBjbFPkWTC3BxGez1NQVgDhXf0OLnC8%2F9zpOpj6Kx5%2BRZv2ISTUlUsEL3eVIHm8A0H%2FSdh4croM%2FChxtOnG9AW4V4OF37%2Fd8sJR0jOAEx4m%2FQZoo1%2FSsHuPyrrmOmmePRR2ljGJ6pYK5e7m9uWQXnGTJcafbxm0xsxHXA9WedF8%2BJs%2FO0N2lkV%2B9IFYgAXyy3E1DZ9s8oLPqAETnEdmCUcbWROmAZUt5U2HdsNk45syUwkdL2F0w%2B9dvv6BTZ9kUh0CcGsebyjVAwP8BCA2BEOJnf5K%2Bd34HmcXiNKQKJMtDXYsrloiEA2n0IWOa4CLCLdKvzT7%2F4adtRodbOb%2F5c2A8Dbu%2FRmCypDmDAHllQ8mrcPp1AyJH%2BdkZtHD3NW1hZM7pZ%2Fq13qxeYuAkOHFR3ACWm0fuZ2w5lkhPLLEXbvhyI%2BO%2BZJIwPp8httUVEYSDuF2pxnaqNJliQgclxPjZiXSn5qzVNwGUh5LFIzC0pYLrumQFcuMqlDplYFr9krcxMQ%3D%3D8bae903309740f869d3ee5bd97474adcb87b9155; bm_sv=74A7C91643E6AB218F20A721C7A690B1~1Swb2lp+mCO0C/Am4qbl1KajUxki7p/gAoKukKPgCCaHM5i5StAfgCpiNWtRNA2A1rxob+mkyj+gi+xxVxL1GXZJIC86GJEOg5VNj7xGJ4tdvX88no5eb68fwPGyVu25QUQUlVZ3+Cix1Y4V+yCXCncvDLrUpoYZHCD3XKedKYc='; ...
    'referer' 'https://m.bithumb.com/trade/chart/BTC'; ...
    'sec-fetch-mode' 'cors'; ...
    'sec-fetch-site' 'same-origin'; ...
    'x-requested-with' 'XMLHttpRequest'; ...
    'User-Agent' 'Mozilla/5.0 (Windows NT 5.1; rv:19.0) Gecko/20100101 Firefox/19.0'});
flag=1;
while flag==1
    try
        url=sprintf('https://m.bithumb.com/trade_history/chart_data?_=%d',b_code.h_t);
        temp=webread(url, ...
            'coinType','C0101', ...
            'crncCd','C0100', ...
            'tickType','24H', ...
            'csrf_xcoin_name','0a27da1a3f215e7f634f6e5682738c91', ...
            options);
        flag=0;
    catch
        fprintf('re try webread\n');
        flag=1;
    end
end
temp=jsondecode(temp);
temp_data=temp.block1;