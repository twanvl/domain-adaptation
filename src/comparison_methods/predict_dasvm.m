function [y_tgt,ys,models] = predict_dasvm(x_src, y_src, x_tgt, varargin)
  % Based on
  % Domain Adaptation Problems, A DASVM Classification Technique and a Circular Validation Strategy
  % Lorenzo Bruzzone, Mattia Marconcini
  % Transactions on Pattern Analysis and Machine Intelligence, vol 32, no 5, pp 770-787, may 2010
  
  % Note: only works for two class problems, use one vs all for multi class problems
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'type') opts.type = 3; end
  if ~isfield(opts,'C') opts.C = 1; end
  if ~isfield(opts,'use_source_C') opts.use_source_C = false; end
  if ~isfield(opts,'bias') opts.bias = false; end
  if ~isfield(opts,'rho') opts.rho = ceil(0.05 * size(x_tgt,1)); end % number of points to add per class in each iteration
  if ~isfield(opts,'beta') opts.beta = 3e-2; end % stopping condition
  if ~isfield(opts,'num_folds'), opts.num_folds = 3; end % for use_source_C
  
  if nargout >= 2
    ys = [];
  end
  if nargout >= 3
    models = {};
  end
  
  % Parameters
  rho = opts.rho;
  
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
  end
  
  % Phase1: initialization
  %[y_tgt,s_tgt,model] = opts.classifier(x_src, y_src, x_tgt, opts.classifier_opts);
  y_tgt = zeros(size(x_tgt,1),1);
  keep_src = ones(size(y_src))  == 1;
  keep_tgt = zeros(size(y_tgt)) == 1;
  svmopts = sprintf('-q -s %d -c %g -B %d',opts.type, opts.C, ifelse(opts.bias,1,-1));
  
  % Phase2: Iterative Domain Adaptation
  max_iterations = length(y_src)+length(y_tgt)+1;
  for it = 1:max_iterations
    % Train SVM
    model = train([y_src(keep_src,:);y_tgt(keep_tgt,:)], sparse([x_src(keep_src,:); x_tgt(keep_tgt,:)]), svmopts);
    models{it} = model;
    
    % New predictions for target
    [y_tgt,~,s_tgt] = predict(y_tgt, sparse(x_tgt), model,'-q');
    if size(s_tgt,2) == 1
      s_tgt = [s_tgt,-s_tgt];
    end
    ys(:,it) = y_tgt;
    
    % Find points closest to the margin (H)
    H = [];
    nH = zeros(1,size(s_tgt,2));
    for cls=1:size(s_tgt,2)
      in_margin = find(0 <= s_tgt(:,cls) & s_tgt(:,cls) <= 1 & ~keep_tgt);
      [~,idx] = sort(1-s_tgt(in_margin,cls));
      if length(idx) > rho
        idx = idx(1:rho);
      end
      nH(cls) = length(idx);
      H = [H; in_margin(idx)];
    end
    
    % Remove points if semilabel changed
    if it > 1
      S = ys(:,it) ~= ys(:,it-1);
      keep_tgt = keep_tgt & ~S;
    else
      S = [];
    end
    
    % Remove source points far away from separating hyperplane
    [~,~,s_src] = predict(y_src, sparse(x_src), model,'-q');
    if size(s_src,2) == 1
      s_src = [s_src,-s_src];
    end
    kept = find(keep_src);
    Q = [];
    for cls=1:size(s_src,2)
      if isempty(H)
        nH(cls) = rho; % If none of the remaining unlabeled samples fall into the margin band the number of patterns to delete is set to ρ
      end
      [~,idx] = sort(s_src(kept,cls),'descend');
      if length(idx) > nH(cls)
        idx = idx(1:nH(cls));
      end
      Q = [Q; kept(idx)];
    end
    
    % Update
    keep_tgt(H) = 1;
    keep_src(Q) = 0;
    
    % Stopping condions
    if isempty(Q) && length(H) <= ceil(opts.beta * length(y_tgt)) && (it > 1 && nnz(S) <= ceil(opts.beta * length(y_tgt)))
      %warning('Stopping condition reached: %d  %d %d %d', it, length(Q), length(H), nnz(S));
      break;
    end
  end
end

