function y_tgt = predict_dasvm_ova(x_src, y_src, x_tgt, varargin)
  % One vs all prediction for multi class DASVM
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'force_ova') opts.force_ova = false; end
  if ~isfield(opts,'use_source_C') opts.use_source_C = false; end
  if ~isfield(opts,'type') opts.type = 3; end
  if ~isfield(opts,'C') opts.C = 1; end
  if ~isfield(opts,'bias') opts.bias = false; end
  if ~isfield(opts,'num_folds'), opts.num_folds = 3; end % for use_source_C
  
  % cross validation on source domain to choose C
  if opts.use_source_C
    Cs = [0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0 10 100 1000 10000];
    acc = zeros(size(Cs));
    for i = 1:numel(Cs)
      svmopts = sprintf('-q -s %d -c %g -B %d -v %d',opts.type, Cs(i), ifelse(opts.bias,1,-1), opts.num_folds);
      acc(i) = train(y_src, sparse(x_src), svmopts);
    end
    [best_acc,best_i] = max(acc);
    opts.C = Cs(best_i);
    opts.use_source_C = false; % don't do cross validation for each one-vs-all classifier
  end
  
  x_src = sparse(x_src);
  x_tgt = sparse(x_tgt);
  y_tgt = zeros(size(x_tgt,1), 1);
  labels = unique(y_src);
  if numel(labels) == 2 && !opts.force_ova
    i_tgt = predict_dasvm(x_src, double(y_src == labels(2)), x_tgt, opts);
    y_tgt = labels(i_tgt + 1);
  else
    s_tgt = zeros(size(x_tgt,1), numel(labels));
    for i=1:numel(labels)
      [~,~,models] = predict_dasvm(x_src, double(y_src == labels(i)), x_tgt, opts);
      model = models{end};
      [~,~,s_tgt_i] = predict(y_tgt, x_tgt, model, '-q');
      if model.Label(1) == 0
        s_tgt_i = -s_tgt_i;
      end
      s_tgt(:,i) = s_tgt_i;
    end
    [s_tgt_max,i_tgt] = max(s_tgt,[],2);
    y_tgt = labels(i_tgt);
  end
end

