% build timing model for ISCAS85
% using simple logical effort model
% and use lsgs to optimize gate sizes

% Example distributed with the
% LSGS Large-scale Gate Sizing MATLAB Toolbox.
% Copyright 2007 Siddharth Joshi and Stephen Boyd. 
% See the file `copyright' (included in the distribution) 
% for full copyright information.

n = 6; %number of gates
dmin =   [2.3310    0.9990    0.9990    0.9990    1.9980    1.9980]';
g = [1.4005    1.3484    1.0100    1.9427    8.3406    9.1768]';

F = sparse(n,n);
F(2,3) = 0.9990;
F(2,4) = 0.9990;
F(1,5) = 1.3320;
F(3,5) = 1.3320;
F(3,6) = 1.6650;
F(4,6) = 1.6650;

a =  [16 3 3 3 8 10]'; % area

% Circuit delay for minimum area: 20.526
% Minimum circuit delay (Tmin): 4.329
% timing specification  
T = 12; 
fprintf('\nSizing ISCAS85 c17 circuit for timing specification T = %.2f\n', T);
x = lsgs(a, g, F, dmin, T);
fprintf('Optimal gate sizes are:\n')
disp(x)

fprintf('\nOptimizing ISCAS85 c17 circuit with all intial gate sizes = 2\n');
x0 = 2*ones(n,1);
[x, t, d, cumpcgiters, areahard, areasoft] = lsgs(a, g, F, dmin, [], x0);
fprintf('\nOriginal area (all gates sizes = 2): %.3f\nOptimal area for the same circuit delay: %.3f\n', a'*x0, a'*x);
fprintf('Area reduction ratio ((original-optimal)/original): %.3f\n', (a'*x0 - a'*x)/(a'*x0));
fprintf('Optimal gate sizes are:\n');
disp(x)



