function y_tgt = predict_flda_cv(x_src, y_src, x_tgt, varargin)
  % Feature Level Domain Adaptation
  % 
  % Use cross-validation on the source domain to pick lambda
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'lambdas'), opts.lambdas = 10.^(-4:0.5:5); end
  if ~isfield(opts,'distribution'), opts.distribution = 'dropout'; end % dropout or blankout
  if ~isfield(opts,'loss'), opts.loss = 'log'; end % qd or log
  if ~isfield(opts,'num_folds'), opts.num_folds = 3; end
  
  fold_idx = randi(opts.num_folds,size(y_src,1),1);
  acc = zeros(size(opts.lambdas));
  for i=1:numel(opts.lambdas)
    for fold=1:opts.num_folds
      in_fold = fold_idx == fold;
      y = predict_flda(x_src(in_fold,:), y_src(in_fold,:), x_src(~in_fold,:), ...
          'distribution',opts.distribution, 'loss',opts.loss, 'lambda',opts.lambdas(i));
      acc(i) = acc(i) + sum(y == y_src(~in_fold,:));
    end
  end
  
  [~,i] = max(acc);
  best_lambda = opts.lambdas(i);
  best_lambda
  y_tgt = predict_flda(x_src, y_src, x_tgt, ...
          'distribution',opts.distribution, 'loss',opts.loss, 'lambda',best_lambda);
end
