function y_tgt = predict_flda(x_src, y_src, x_tgt, varargin)
  % Feature Level Domain Adaptation
  % 
  % Based on code from https://github.com/wmkouw/da-fl
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'lambda'), opts.lambda = 1; end
  if ~isfield(opts,'distribution'), opts.distribution = 'dropout'; end % dropout or blankout
  if ~isfield(opts,'loss'), opts.loss = 'log'; end % qd or log
  
  addpath('src/comparison_methods/da-fl');
  fun = sprintf('flda_%s_%s', opts.loss, opts.distribution);
  [W,theta] = feval(fun, x_src', x_tgt', y_src, opts.lambda);
  % W is num_features*num_classes
  classes = unique(y_src);
  if numel(classes) == 2
    y_tgt = (x_tgt * W(:,1)) > 0;
  else
    [~,y_tgt_idx] = max(x_tgt * W, [], 2);
    y_tgt = classes(y_tgt_idx);
  end
end
