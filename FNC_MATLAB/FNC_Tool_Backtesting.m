function [Balance,CAGR_day,CAGR_year,MDD]=FNC_Tool_Backtesting(P_Price, P_Signal)
%% load sample
%{
clc;
close all;
clearvars;
load('../../DATA/FNC/KRX.mat');
index = 851;
P_Price = KRX(index).Price;
P_Signal = KRX(index).PST;
%}

%% Variable initailize
Price(1) = P_Price(1);
Signal(1) = P_Signal(1);
Price = [Price; P_Price];
Signal = [Signal; P_Signal];

%% Parmeter Setting
State=0; % non-hold
buy_price=0;
Balance(1)=1;

buy_tax=0;
sell_tax=0.003; % 0.3%

%% Balacne
for i=2:length(Signal)
    if State == 0 % non-hold
        Balance(i)=Balance(i-1);
        if Signal(i) == -1 % buy
            State = 1;
        end
    elseif State == 1 % hold
        rate= (Price(i) - Price(i-1)) / Price(i-1);
        Balance(i) = Balance(i-1) + (Balance(i-1) * rate);
        if Signal(i) == 1 % sell
            State = 0;
            Balance(i) = Balance(i) - (Balance(i)*sell_tax);
        end
    end
end

%% CAGR, MDD 
CAGR_day = ((Balance(length(Balance)) / Balance(1)) ^ (1/(length(Balance)))) - 1;
CAGR_year = ((Balance(length(Balance)) / Balance(1)) ^ (1/(length(Balance)/365))) - 1;
MDD = min(Balance) - Balance(1) / Balance(1);
fprintf('CAGR_day = %f%%\nCAGR_year = %f%%\nMDD = %f%%\n', CAGR_day*100, CAGR_year*100, MDD*100);
% figure; plot(Price); figure; plot(Balance);