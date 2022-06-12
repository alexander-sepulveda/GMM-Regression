function [E_YX] = GMM_prediction(mix, X, Nx, Ny, Nmix, Cov_type)

% It predicts Y (the output) from data in X (matrix of inputs), using GMM
% regression in mix. It is based on the NetLab toolbox.
% Inputs:
%        - mix: GMM model for Z = [X, Y].
%        - X  : Input data, where each row corresponds to an observation.
% Outputs:
%        - expected value of Y given X.
% Procedure:
% E(Y/X) = \sum_{j=1}^K  B_j(X)*m_j(X)    ; K = Nmix variable.
%    m_j(X) = \mu_j^Y + C_j^{YX} * inv(C_j^X) * (X - \mu_j^X) 
%    B_j(X) : responsabilities or, posterior probabilities of j-mixture
%             given input X. See 2.192 equation of Pattern Recognition
%             and Machine Learning book by C. Bishop.

addpath('/home/alexander/TOOLS/Netlab/');   % path to Netlab Toolbox
Ndata = length(X(:, 1));     % number of observation in data.

%-- take it out those components from the mixture model corresponding to inputs X.
mix_X = gmm(Nx, Nmix, Cov_type);   % Creates a GMM with an architecture according to inputs X.
mix_X.centres = mix.centres(:,1:Nx); % takes it out the means
mix_X.priors  = mix.priors;          % takes it out the mixture weigths
ndim = length(size(mix.covars));     % to know the dimension type of covariance matrix 
if (ndim==3)       % COV is a full matrix.
   mix_X.covars = mix.covars(1:Nx,1:Nx,:);   
end
if (ndim==2)       % COV is a diagonal matrix.
   mix_X.covars = mix.covars(:,1:Nx);   
end

%--- calculate responsabilities (posterior probabilities of j-mix given X).
%post = gmmactiv(mix_X, X );
%nume = post.*repmat(mix_X.priors, Ndata, 1);
%deno = post*(mix_X.priors');
%resp = nume./ repmat(deno, 1, mix_X.ncentres);
resp = gmmpost(mix_X, X);   % gmmpost replace the lines just before.

E_YX = zeros(Ndata, 1);
for(j=1:Nmix)
   % take it out covariance matrices.
   if (ndim==3)       % COV is a full matrix.
      Cov_mat = mix.covars(:, :, j); 
      CovXX = Cov_mat(1:Nx,1:Nx);
      CovYY = Cov_mat(Nx+1:end,Nx+1:end);
      CovXY = Cov_mat(1:Nx,Nx+1:end);
      CovYX = Cov_mat(Nx+1:end,1:Nx);
   end
   if (ndim==2)       % COV is a diagonal matrix.
      Cov_mat = mix.covars;
      CovXX = diag(Cov_mat(j, 1:Nx), 0);
      CovYY = diag(Cov_mat(j, Nx+1:end), 0);
      CovXY = zeros(Nx, Ny);
      CovYX = zeros(Ny, Nx);
   end
   % estimate m_j(X)
   muX  = mix.centres(j, 1:Nx);    muY = mix.centres(j, Nx+1:end);
   M_jX =  CovYX*inv(CovXX)*transpose((X - repmat(muX, Ndata, 1))) ;
   M_jX =  repmat(muY, Ndata, 1) + transpose(M_jX);
   
   % estimate Ej_YX
   E_YX = E_YX +  resp(:, j).*M_jX;
end

