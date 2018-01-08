function loss = tsvm_loss(x_src,y_src, x_tgt,y_tgt, model, varargin)
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'type'), opts.type = 3; end
  if ~isfield(opts,'C'), opts.C = 1; end
  if ~isfield(opts,'balanced'), opts.balanced = false; end
  
  if nargin<5 || isempty(model)
    model = train([y_src;y_tgt],sparse([x_src;x_tgt]),sprintf('-q -s %d -c %f',opts.type,opts.C));
  end
  
  n_src = size(x_src,1);
  n_tgt = size(x_tgt,1);
  
  % Find w*x*y
  if isempty(y_tgt)
    y_tgt = zeros(n_tgt,1);
    [y,~,s] = predict([y_src;y_tgt], sparse([x_src;x_tgt]), model,'-q');
    y_tgt = y(n_src+1:end,:);
  else
    [~,~,s] = predict([y_src;y_tgt], sparse([x_src;x_tgt]), model,'-q');
  end
  if size(s,2)==1
    s = [s,-s];
  end
  y = [y_src;y_tgt];
  y_idx = lookup(model.Label, y);
  idx = sub2ind(size(s), (1:size(s,1))', y_idx);
  wxy = s(idx);
  
  if opts.balanced
    % class weights for target part
    indicator = sparse(1:size(s,1), y_idx(n_src+1:end), 1, size(y,1),length(model.Label));
    class_size = full(sum(indicator));
    cw = n_tgt ./ (class_size + 1e-2) / length(model.Label);
    w = [repmat(1,n,1); cw(y_idx(n_src+1:end))];
  else
    w = 1;
  end
  
  % Form of objective used by liblinear, see http://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
  loss = opts.C * sum(w .* max(0,1-wxy)) + 0.5 * sum(sum(model.w.^2));
  if opts.balanced
    loss = loss + sum(cw) * 1e-2;
  end
end
