%Trabalho 1 - Redes Neurais - UFRGS
%Classificador BP

clear all; close all; clc;

%Carregar os arquivos com os dados
load ecoli_inputs;
ecoli_inputs = ecoli_inputs';
load ecoli_targets;
ecoli_targets = ecoli_targets';

%total de dados do problema:
tam = length(ecoli_inputs);

%Criar a rede neural 
hiddenLayerSize =  [10];
net = patternnet(hiddenLayerSize);
net.trainFcn = 'trainlm';

net.trainParam.mu = 0.001

%Divisão dos dados em Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 10/100;

%Validação Cruzada LOO
net.trainParam.showWindow = false;
for i = 1:336
    inLOO = ecoli_inputs;
    outLOO = ecoli_targets;

    %Salva a entrada que vai ser testada
    inputTeste = inLOO(:,i);
    outputTeste = outLOO(:,i);
    
    %Remove uma entrada
    inLOO(:,i) = [];
    outLOO(:,i) = []; 
    
    %Treina a rede
    [net,tr] = train(net,inLOO,outLOO);
    
    %Testa com a entrada deixada de fora
    outputs(:,i) = net(inputTeste);
end

%Gerar a matriz de confusão
[c,cm,ind,per] = confusion(ecoli_targets,outputs);