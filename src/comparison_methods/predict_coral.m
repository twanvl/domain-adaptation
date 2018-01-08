function Ytt = predict_coral(Xr,Yr,Xtt, varargin)
  % CORAL
  %
  % Code adapted from https://github.com/VisionLearningGroup/CORAL
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  
  % don't run on high dimensional data
  if size(Xr,2) > 10000
    Ytt = [];
    return
  end
  
  cov_source = cov(Xr) + eye(size(Xr, 2));
  cov_target = cov(Xtt) + eye(size(Xtt, 2));
  A_coral = cov_source^(-1/2)*cov_target^(1/2);
  Sim_coral = Xr * A_coral * Xtt';
  
  Ytt = zeros(size(Xtt,1),1);
  Ytt = SVM_predict(Xr, A_coral, Ytt, Sim_coral, Yr);
end

function predicted_label_values = SVM_predict(trainset, M,testlabelsref,Sim,trainlabels)
  Sim_Trn = trainset * M *  trainset';
  index = [1:1:size(Sim,1)]';
  Sim = [[1:1:size(Sim,2)]' Sim'];
  Sim_Trn = [index Sim_Trn ];

  C = [0.001 0.01 0.1 1.0 10 100 1000 10000];
  parfor i = 1:size(C,2)
    model(i) = svmtrain(trainlabels, double(Sim_Trn), sprintf('-t 4 -c %g -v 2 -q',C(i)));
  end
  [val indx]=max(model);
  CVal = C(indx);
  model = svmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %g -q',CVal));
  [predicted_label_values] = svmpredict(testlabelsref, Sim, model);
end

