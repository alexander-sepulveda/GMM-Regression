% programa para realizar una regresión usando GMM para un problema sencillo.

addpath('/home/alexander/DOCENCIA/APRENDIZAJE_AUTOMATICO/LABS/GMM_regression/Netlab/');
% configure GMM training.
OPTIONS(1)  = -1; % Switch off all messages, including warning
OPTIONS(3)  = 0.001;   
OPTIONS(5)  = 1;
OPTIONS(14) = 25;   % number of iterations.

Cov_type = 'full';   % type of covariance matrix: 'diag', 'full', 'spherical'

% ---------- leyendo los datos.
name_file_xls = 'Advertising.csv'; % data used in the book: 'Introduction to Statistical Learning' by T. Hastie et. al.
A = readtable(name_file_xls);     [N_filas, N_cols] = size(A);
TV = table2array( A(:, 'TV') );   radio = table2array( A(:, 'radio') );  newspaper = table2array( A(:, 'newspaper') );
Sales = table2array( A(:, 'sales') );   Y = Sales;
%X = [TV, radio, newspaper];
X = [TV];
X = zscore(X);     Y = zscore(Y);

N_inputs = min(size(X));         N_outputs = min(size(Y));
% --------------------------------
N_mixtures = 3;
%-------- estimate GMM from data.
mix = gmm(N_inputs+N_outputs, N_mixtures, Cov_type);  % Creates a Gaussian mixture model (dimension, # mixtures, type of cov matrix).
mix = gmminit(mix, [X , Y], OPTIONS);  % Initialises GMM from data. The k-means algorithm is used to determine the centres.
mix = gmmem(mix, [X , Y], OPTIONS);

mix

E_YX = GMM_prediction(mix, X, N_inputs, N_outputs, N_mixtures, Cov_type);

plot(E_YX, 'b')
hold on
plot(Y, 'k')
hold off
