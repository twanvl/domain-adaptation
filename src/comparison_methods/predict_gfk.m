function y_tgt = predict_gfk(x_src, y_src, x_tgt, varargin)
  % GFK
  % See http://www-scf.usc.edu/~boqinggo/domainadaptation.html
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'use_pls'), opts.use_pls = false; end;
  if ~isfield(opts,'d'), opts.d = 10; end;
  if ~isfield(opts,'svm'), opts.svm = false; end;
  if ~isfield(opts,'svm_sqrt'), opts.svm_sqrt = false; end;
  if ~isfield(opts,'skip_large_feature_space'), opts.skip_large_feature_space = true; end;
  
  if opts.skip_large_feature_space && size(x_src,2) > 10000
    y_tgt = [];
    return
  end
  
  % Do PCA on X
  if opts.use_pls
    Ps = plsregress(x_src);
  else
    Ps = princomp(x_src);
  end
  Pt = princomp(x_tgt);
  
  addpath('src/comparison_methods/ToRelease_GFK')
  G = GFK([Ps,null(Ps')], Pt(:,1:opts.d));
  
  if opts.svm
    if opts.svm_sqrt
      %sqG = sqrtm(G);
    	A = chol(G + eps*20*eye(size(G,1)));
    	sqG = A';
      y_tgt = predict_liblinear_cv(x_src * sqG, y_src, x_tgt * sqG);
    else
      % based on coral.m
      Sim = x_src * G * x_tgt';
      y_tgt = zeros(size(x_tgt,1),1);
      y_tgt = SVM_predict(x_src, G, y_tgt, Sim, y_src);
    end
  else
    y_tgt = my_kernel_knn(G, x_src, y_src, x_tgt);
  end
end

function prediction = my_kernel_knn(M, Xr, Yr, Xt)
  dist = repmat(diag(Xr*M*Xr'),1,size(Xt,1)) ...
       + repmat(diag(Xt*M*Xt')',length(Yr),1)...
       - 2*Xr*M*Xt';
  [~, minIDX] = min(dist);
  prediction = Yr(minIDX);
end

function predicted_label_values = SVM_predict(trainset, M,testlabelsref,Sim,trainlabels)
  Sim_Trn = trainset * M *  trainset';
  index = [1:1:size(Sim,1)]';
  Sim = [[1:1:size(Sim,2)]' Sim'];
  Sim_Trn = [index Sim_Trn ];

  flags = '-h 0'; % Needed on some datasets
  C = [0.001 0.01 0.1 1.0 10 100 1000 10000];
  parfor i = 1:size(C,2)
    model(i) = svmtrain(trainlabels, double(Sim_Trn), sprintf('-t 4 -c %g -v 2 -q %s',C(i),flags));
  end
  [val indx]=max(model);
  CVal = C(indx);
  model = svmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %g -q %s',CVal,flags));
  [predicted_label_values] = svmpredict(testlabelsref, Sim, model);
end

