%size 10000 gate circuit using lsgs

% Example distributed with the
% LSGS Large-scale Gate Sizing MATLAB Toolbox.
% Copyright 2007 Siddharth Joshi and Stephen Boyd. 
% See the file `copyright' (included in the distribution) 
% for full copyright information.

clear all
fprintf('\nThis script loads a 10000 gate example circuit and');
fprintf('\nuses lsgs to size it for 5 timing specifications.\n');
fprintf('\nEach sizing should take around 10-20 seconds.\n');
load('ckt10k.mat'); % load data for 10000 gate circuit
fprintf('\nExample circuit with 10000 gates loaded.\n');
T = [100 110 120 130 140]; % a set of timing specifications
fprintf('\nSizing 10000 gate circuit:\n');
area = [];
for i=1:length(T)
    fprintf('\tfor timing specification T = %.3f ...', T(i));
    x = lsgs(a, g, F, dmin, T(i), [], 1, 250);
	area(i) = a'*x;
    fprintf('done. Area = %.3e\n', area(i));
end
fprintf('Done.\n');

plot(T, area);
xlabel('Timing specification');
ylabel('Area');
title('Trade-off curve');
