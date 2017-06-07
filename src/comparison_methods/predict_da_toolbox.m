function y_tgt = predict_da_toolbox(x_src, y_src, x_tgt, ftTrans, params)
  % Perform a prediction with one of the ftTrans functions from the matlab domain adaptation toolbox.
  % See https://github.com/viggin/domain-adaptation-toolbox
  %
  % Usage:
  %    predict_da_toolbox(x_src, y_src, x_tgt, ftTrans, params)
  % Where
  %    ftTrans = name of a feature transformation function
  %    params  = struct of parameters (optional)
  
  if nargin < 5
    params = struct();
  end
  addpath('src/comparison_methods/domain-adaptation-toolbox');
  addpath('src/comparison_methods/domain-adaptation-toolbox/ToRelease_GFK');
  addpath('src/comparison_methods/domain-adaptation-toolbox/ssa');
  
  % Call feature transformation
  % functions look like:
  %   function [ftAllNew,transMdl] = ftTrans(ftAll,maSrc,target,maLabeled,param)
  ftAll = [x_src;x_tgt];
  maSrc = logical([ones(size(x_src,1),1); zeros(size(x_tgt,1),1)]);
  labeled = true;
  if labeled
    target = y_src;
    maLabeled = maSrc;
  else
    target = [];
    maLabeled = zoers(size(ftAll,1));
  end
  x = feval(ftTrans, ftAll,maSrc,target,maLabeled,params);
  x_src = x(1:size(x_src,1),:);
  x_tgt = x(size(x_src,1)+1:end,:);
  
  % Train classifier and predict
  y_tgt = predict_liblinear_cv(x_src, y_src, x_tgt);
end

