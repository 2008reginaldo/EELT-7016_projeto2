%% this script performs an example of all steps of the system identification available  at this repository

clear all
close all
clc


%% steps to perform
identifyModel = true;
identifyNoise =  true;
identifyComplete = true;
computeGFRF = true; % if you do not have the Symbolic Toolbox,set this to false or use the Octave system. The other parts of the system 
                    %identification will work normally
computeNOFRF = true; % if you do not have the Symbolic Toolbox,set this to false or use the Octave system. The other parts of the system 
                    %identification will work normally

%% data

% obtain a signal of the system y(k) = -0.1*y(k-1) - 0.5*u(k-1)*y(k-1) + 0.1*u(k-2). You
% can change it as you wish, or use your own data collected elsewhere.
% The data must be in column format. It is also possible to use in the same
% identification process different data acquisitions from the same system.
% Each data acquisition must be in a different column and the columns from
% the input matrix must be correspondent to the columns from the output
% matrix

Fs = 100; % Sampling frequency of the data acquisition, in Hz. It is used only for the GFRFs and NFRFs computation

u = randn(2000,1);
y = 0.0*ones(size(u));

for k = 5:length(y)
   y(k) = 0.1*y(k-1) - 0.5*u(k-1)*y(k-1) + 0.1*u(k-2); 
end

%% 
input = (u(100:end));  % throw away the first 100 samples to avoid transient effects
output = (y(100:end));  % throw away the first 100 samples to avoid transient effects
mu = 2; % maximal lag for the input. In this case, we know that mu is 2 but normally we do not!
my = 1; % maximal lag for the output. In this case, we know that my is 1 but normally we do not!   
degree = 2; % maximal polynomial degree. In this case, we know that degree is 2 but normally we do not! 
delay = 1; % the number of samplings that takes to the input signal effect the output signal. In this case we know that 
%degree is 2 but normally we do not! 
dataLength = 500; % number of samplings to be used during the identification process. Normally a number between 400 and
% 600 is good. Do not use large numbers.
divisions = 1; % Number of parts of each data acquisition to be used in the identification process
pho = 1e-2; % a lower value will give you more identified terms. A higher value will give you less.
phoL = 1e-2; % a lower value will give you more identified terms. A higher value will give you less. This is only used 
%if you want to compute the GFRFs, to guarantee that at least one term will be linear. In this case, change the variable
%flag in %NARXModelIdentificationOf2Signals.m  file to 1



%%
if identifyModel    
    [Da, a, la, ERRs] = NARXModelIdentificationOf2Signals(input, output, degree, mu, my, delay, dataLength, divisions, ...
       pho);
    %%
    identModel.terms = Da;
    identModel.termIndices = la;
    identModel.coeff = a;
    identModel.degree = degree;
    identModel.Fs = Fs;
    identModel.ESR = 1-ERRs;
    %%    
    save(['testIdentifiedModel' num2str(Fs) '.mat'], 'identModel');
else
    load(['testIdentifiedModel' num2str(Fs) '.mat']);
    Da = identModel.terms;
    la = identModel.termIndices;
    a = identModel.coeff;
    degree = identModel.degree;
    Fs = identModel.Fs;
end

%%
