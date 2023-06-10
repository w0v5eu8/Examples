clearvars;
clc;

%% load
load('../../DATA/FNC/KRX_Index.mat');
load('../../DATA/FNC/KRX.mat');
load('../../DATA/FNC/ETF.mat');

%% KRX_Index
ticker_i = 1;

if KRX_Index(ticker_i).nPST == 0
    fprintf('There is not PST\n');
    return;
else
    fprintf('PST: %s, nPst = %d\n', KRX_Index(ticker_i).codeName, KRX_Index(ticker_i).nPST);
end

% plot -1(buy), 1(sell) point => (x,y) with Price price
x=[];
y=[];
index=0;
for i=1:length(KRX_Index(ticker_i).PST)
    if (KRX_Index(ticker_i).PST(i) == 1 || KRX_Index(ticker_i).PST(i) == -1)
        index=index+1;
        x(index)=i;
        y(index)=KRX_Index(ticker_i).Price(i);
    end
end

figure;
plot(KRX_Index(ticker_i).Price);
hold on;
plot(x,y,'r');
axis('tight');
fprintf('length = %d\n',length(x));

%% KRX
ticker_i = 851;

if KRX(ticker_i).nPST == 0
    fprintf('There is not PST\n');
    return;
else
    fprintf('PST: %s, nPst = %d\n', KRX(ticker_i).codeName, KRX(ticker_i).nPST);
end

% plot -1(buy), 1(sell) point => (x , y) with Price price
x=[];
y=[];
index=0;
for i=1:length(KRX(ticker_i).PST)
    if (KRX(ticker_i).PST(i) == 1 || KRX(ticker_i).PST(i) == -1)
        index=index+1;
        x(index)=i;
        y(index)=KRX(ticker_i).Price(i);
    end
end

figure;
plot(KRX(ticker_i).Price);
hold on;
plot(x,y,'r');
axis('tight');
fprintf('length = %d\n',length(x));

%% ETF
ticker_i = 1;

if ETF(ticker_i).nPST == 0
    fprintf('There is not PST\n');
    return;
else
    fprintf('PST: %s, nPst = %d\n', ETF(ticker_i).codeName, ETF(ticker_i).nPST);
end

% plot -1(buy), 1(sell) point => (x , y) with Price price
x=[];
y=[];
index=0;
for i=1:length(ETF(ticker_i).PST)
    if (ETF(ticker_i).PST(i) == 1 || ETF(ticker_i).PST(i) == -1)
        index=index+1;
        x(index)=i;
        y(index)=ETF(ticker_i).Price(i);
    end
end

figure;
plot(ETF(ticker_i).Price);
hold on;
plot(x,y,'r');
axis('tight');
fprintf('length = %d\n',length(x));

%{
A1
PST: »ï¼ºÀüÀÚ, nPst = 6180
length = 539
PST: KODEX 200, nPst = 4200
length = 569

A2
PST: »ï¼ºÀüÀÚ, nPst = 6180
length = 621
PST: KODEX 200, nPst = 4180
length = 303

A3
PST: »ï¼ºÀüÀÚ, nPst = 6201
length = 505
PST: KODEX 200, nPst = 4200
length = 527
%}