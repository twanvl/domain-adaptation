function y_tgt = predict_sa(x_src, y_src, x_tgt, varargin)
  % Subspace alignment
  % 
  % Adapted from http://users.cecs.anu.edu.au/~basura/DA_SA/
  % by Basura Fernando
  % 
  % @inproceedings{Fernando2013b,
  % author = {Basura Fernando, Amaury Habrard, Marc Sebban, Tinne Tuytelaars},
  % title = {Unsupervised Visual Domain Adaptation Using Subspace Alignment},
  % booktitle = {ICCV},
  % year = {2013},
  % } 
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'svm_sqrt'), opts.svm_sqrt = false; end;
  if ~isfield(opts,'subspace_dim'), opts.subspace_dim = 80; end;
  
  [Xss,~,~] = princomp(x_src);
  [Xtt,~,~] = princomp(x_tgt);
  Xs = Xss(:,1:opts.subspace_dim);
  Xt = Xtt(:,1:opts.subspace_dim);

  A = (Xs*Xs')*(Xt*Xt');
  if opts.svm_sqrt
  	C = chol(A + eps*200*eye(size(A,1)));
  	sqA = C';
    y_tgt = predict_liblinear_cv(x_src * sqA, y_src, x_tgt * sqA);
  else
    % this is used in the original source code
    Sim = x_src * A *  x_tgt';
    y_tgt = zeros(size(x_tgt,1),1);
    y_tgt = SVM_predict(x_src, A, y_tgt, Sim, y_src);
  end
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

