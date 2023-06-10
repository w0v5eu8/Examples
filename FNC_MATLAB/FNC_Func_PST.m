function [out_arg1, out_arg2] = FNC_Func_PST(T,P,c)
%{
close all;
clear all;
clc;
load ('../../DATA/FNC/KRX.mat');
c=DATA(2651).Close;
T=5;
P=0.05;
%}

%% variable setting
c=double(c);
n=length(c);

%% critical point: cp
% find local minima, maxima
[ymax,xmax,ymin,xmin] = FNC_Func_Extrema(c);
cp1=vertcat(xmax,xmin);
cp2=vertcat(ymax,ymin);
[~, b]=sort(cp1);
cp=[cp1(b) cp2(b)];

if isempty(cp)
    out_arg1=[];
    out_arg2=0;
    return;
end

%% smoothing point: sp
index=1;
sp(index,1:2)=cp(index,1:2);

i=2;
while i<=length(cp)-1
    if ( cp(i+1,1)-cp(i,1) < T ) &&  (( abs(cp(i+1,2)-cp(i,2)) / ((cp(i,2)+cp(i+1,2))/2) ) < P)
        i=i+2;
        continue;
    else
        index=index+1;
        sp(index,1)=cp(i,1);
        sp(index,2)=cp(i,2);
        i=i+1;
    end
end

if size(sp,1) == 1
    %% if smoothing point(sp) is just one
    % it means can't find BUY and SELL point by PST
    out_arg1(1,1:n) = zeros(1,n);
    out_arg2 = 0;
else
    %% reversal point: rp
    % just remain reversal points
    temp=sp(:,2);
    [ymax,xmax,ymin,xmin] = FNC_Func_Extrema(temp);
    t1=vertcat(xmax,xmin);
    t2=vertcat(ymax,ymin);
    [~, b]=sort(t1);
    rp_t = [sp(t1(b),1) t2(b)];
    
    if isempty(rp_t)
        out_arg1=[];
        out_arg2=0;
        return
    end
    
    % range [-1 1]
    rp(1,1)=rp_t(1,1);
    rp(2,1)=rp_t(2,1);
    if rp_t(1,2) < rp_t(2,2)
        rp(1,2)=-1;
        rp(2,2)=1;
    else
        rp(1,2)=1;
        rp(2,2)=-1;
    end
    for i=3:length(rp_t)
        rp(i,1)=rp_t(i);
        if rp_t(i-1,2) < rp_t(i,2)
            rp(i,2)=1;
        else
            rp(i,2)=-1;
        end
    end
    
    %% output time series result: out
    % drawing line reversal point to reversal point using polyfit
    for i=1:length(rp)-1
        a=polyfit(rp(i:i+1,1),rp(i:i+1,2),1);
        for j=rp(i,1):rp(i+1,1)
            out_arg1(j,1)=j*a(1)+a(2);
        end
    end
    
    for i=rp(length(rp),1)+1:n
        out_arg1(i,1)=0;
    end
    
    % polyfit result is approximate 1 and -1 (it is -1.000~, 1.0000), so it correct just -1 and 1
    out_arg1(rp(:,1))=rp(:,2);
    
    % number of specific reversal pionts.
    out_arg2=rp(length(rp),1);
end