function [y_tgt,opts,model] = predict_liblinear(x_src,y_src,x_tgt, varargin)
  % Train a linear SVM, and use it to predict the labels of a test set.
  % Uses the liblinear svm implementation
  % 
  % Usage:
  %   [y_test] = predict_liblinear(x_train, y_train, x_test, [options])
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'type') opts.type = 3; end
  if ~isfield(opts,'C') opts.C = 1; end
  if ~isfield(opts,'bias') opts.bias = false; end
  if opts.bias
    bias = 1;
  else
    bias = -1; % liblinear default
  end
  
  model = train(y_src, sparse(x_src), sprintf('-q -s %d -c %g -B %d',opts.type, opts.C, bias));
  y_tgt = zeros(size(x_tgt,1),1);
  if nargout>1
    [y_tgt,acc,s_tgt] = predict(y_tgt, sparse(x_tgt), model,'-q');
  else
    y_tgt = predict(y_tgt, sparse(x_tgt), model,'-q');
  end
end

