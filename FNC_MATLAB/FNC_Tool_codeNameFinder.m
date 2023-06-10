function output = FNC_Tool_codeNameFinder(Type,key)
% FNC_Tool_codeNameFinder('KRX','삼성전자')

%% 데이터 로딩
cmd=sprintf('load(''../../DATA/FNC/%s.mat''); DATA=%s;', Type, Type);
eval(cmd);

%%
for i=1:length(DATA)
    if strcmp(DATA(i).codeName, key)
        output = i;
        fprintf('Tyep: %s, codeName: %s, Index: %d\n', Type, key, i);
    end
end