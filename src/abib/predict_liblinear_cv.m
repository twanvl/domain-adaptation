function [y_tgt, best_opts, model] = predict_liblinear_cv(x_src,y_src,x_tgt, varargin)
  % Train a linear SVM, and use it to predict the labels of a test set.
  % Uses the liblinear svm implementation.
  % Uses cross-validation to pick C parameter.
  % 
  % Usage:
  %   [y_test, best_opts, model] = predict_liblinear_cv(x_train, y_train, x_test, [options])
  % 
  % Options:
  %   'type',i       use the given classifier type (see liblinear documentation)
  %   'num_folds',i  number of folds for cross-validation (default 2)
  %   'verbose',i    verbosity (default 0)
  %   'C',cs         list of values of the C parameter to try
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'type'), opts.type = 3; end
  if ~isfield(opts,'C'), opts.C = [0.001 0.01 0.1 1.0 10 100 1000 10000]; end
  if ~isfield(opts,'num_folds'), opts.num_folds = 2; end
  if ~isfield(opts,'verbose'), opts.verbose = 0; end
  if ~isfield(opts,'probability') opts.probability = false; end
  if ~isfield(opts,'bias') opts.bias = false; end
  if opts.bias
    bias = 1;
  else
    bias = -1; % liblinear default
  end
  
  acc = zeros(size(opts.C));
  for i = 1:numel(opts.C)
    acc(i) = train(y_src, sparse(x_src), sprintf('-q -s %d -c %g -B %d -v %d',opts.type,opts.C(i),bias,opts.num_folds));
  end
  [best_acc,best_i] = max(acc);
  best_C = opts.C(best_i);
  best_opts = struct('C', best_C, 'type', opts.type, 'bias', opts.bias, 'probability',opts.probability);
  
  if opts.verbose
    fprintf('[best C: %g]', best_C);
  end
  
  model = train(y_src, sparse(x_src), sprintf('-q -s %d -c %g -B %d',opts.type,best_C,bias));
  y_tgt = zeros(size(x_tgt,1),1);
  if nargout>1
    [y_tgt,acc,s_tgt] = predict(y_tgt, sparse(x_tgt), model,'-q');
  else
    y_tgt = predict(y_tgt, sparse(x_tgt), model,'-q');
  end
end

