function [x_src,x_tgt] = preprocess(x_src, y_src, x_tgt, type)
  % This function applies preprocessing to the features.
  % Preprocessing is applied to source and target data simultaniously
  % 
  % Usage:
  %   [x_src,x_tgt] = preprocess(x_src, y_src, x_tgt, type);
  % 
  % Possible types (combinators):
  %   'none':         no preprocessing
  %   'joint-TYPE':   apply type to source and target together, instead of separately
  %   'TYPE1,TYPE2':  apply type1 followed by type2
  %   
  % Possible types (actual preprocessing):
  %   'std':          divide each feature by its standard deviation
  %   'zscore':       subtract mean and divide by standard deviation for each feature
  %   'norm-row':     normalize each *row* to have sum=1
  %   'truncate':     apply max(0,x), i.e. keep only positive values
  %   'binary':       apply (x > 0.5), i.e. binarize feature values.
  
  if any(type==',')
    comma = find(type==',',1);
    [x_src,x_tgt] = preprocess(x_src, y_src, x_tgt, type(1:comma-1));
    [x_src,x_tgt] = preprocess(x_src, y_src, x_tgt, type(comma+1:end));
    return
  end
  
  if length(type)>3 && isequal(type(1:4),'top-')
    % feature selection: largest total value
    n = sscanf(type(5:end),'%d');
    score = full(sum([x_src;x_tgt]));
    [~,idx] = sort(score,'descend');
    keep = idx(1:n);
    x_src = x_src(:,keep);
    x_tgt = x_tgt(:,keep);
  elseif length(type)>3 && isequal(type(1:4),'svd-')
    % feature aggregation: PCA-like
    n = sscanf(type(5:end),'%d');
    [u,s,v] = svds([x_src;x_tgt],n);
    x_src = u(1:size(x_src,1),:);
    x_tgt = u(size(x_src,1)+1:end,:);
  elseif length(type)>6 && isequal(type(1:6),'joint-')
    % joint
    type = type(7:end);
    x = preprocess_one([x_src;x_tgt], type);
    x_src = x(1:size(x_src,1),:);
    x_tgt = x(size(x_src,1)+1:end,:);
  elseif isequal(type,'std-per-class')
    % within class standard deviation on the source domain
    std_x = sparse_std(x_tgt); % plus a bit of the target, to prevent div by 0 and friends
    total = 1;
    classes = unique(y_src);
    for i=1:numel(classes)
      idx = find(y_src == classes(i));
      std_x = std_x + (length(idx)-1) * sparse_std(x_src(idx,:));
      total = total + length(idx)-1;
    end
    std_x = std_x / total;
    scale = 1 ./ (eps + std_x);
    x_src = mul(x_src,scale);
    x_tgt = mul(x_tgt,scale);
  else
    % separate
    x_src = preprocess_one(x_src, type);
    x_tgt = preprocess_one(x_tgt, type);
  end
end

function x = preprocess_one(x, type)
  if isequal(type,'none')
    % do nothing
  elseif isequal(type, 'std')
    % divide by standard deviation
    scale = 1 ./ (eps + sparse_std(x));
    x = mul(x,scale);
  elseif isequal(type, 'std+0.1')
    % divide by standard deviation
    scale = 1 ./ (0.1 + sparse_std(x));
    x = mul(x,scale);
  elseif isequal(type, 'mean')
    % divide by mean
    scale = 1 ./ (eps + mean(x,1));
    x = mul(x,scale);
  elseif isequal(type, 'zscore')
    % subtract mean, divide by standard deviation
    x = zscore(x,1);
  elseif isequal(type, 'norm-row')
    % divide by row-wise sum
    x = bsxfun(@rdivide, x, sum(x,2));
  elseif isequal(type, 'truncate')
    % truncate negative values
    if issparse(x)
      [a,b,c] = find(x);
      x = sparse(a(c>0), b(c>0), c(c>0), size(x,1), size(x,2));
    else
      x = max(0, x);
    end
  elseif isequal(type, 'log1p')
    % log(1+x)
    x = log1p(x);
  elseif isequal(type, 'binary')
    % binarize with threshold 0.5
    x = double(x > 0.5);
  elseif isequal(type, 'binary-mean')
    % binarize with threshold = mean
    %x = double(bsxfun(@gt, x, mean(x,1)));
    [i,j,v] = find(x);
    mean_x = mean(x,1);
    v = double(v > full(mean_x(j)'));
    x = sparse(i,j,v, size(x,1), size(x,2));
  else
    error('Unknown preprocessing type: %s', type);
  end
end


function c = mul(a,b)
  % c = a * diag(b) = a * repmat(b,size(a,1),1)
  if exist('OCTAVE_VERSION','builtin')
    c = a * diag(b);
  else
    c = bsxfun(@times, a, b);
  end
end

function s = sparse_std(x)
  % Calculate standard deviation, but don't blow up for sparse matrices
  if issparse(x)
    s = sqrt((mean(x.^2,1) - mean(x,1).^2) * size(x,1)/(size(x,1)-1));
  else
    s = std(x);
  end
end

