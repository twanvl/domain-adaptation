function es = majority_votes(ys)
  % Given predictions for each iteration,
  % output prediction of ensemble (majority vote) for each prefix:
  % Usage:
  %   es = majority_votes(ys)
  % Means:
  %   es(i,j) = mode(ys(i,1:j))
  % Except that the behavior is slightly different in the case of ties.
  % With ties, mode prefers lower class labels
  % we instead give the 'previous' answer, i.e. mode(ys(i,1:j-1))
  
  % This implementation keeps a running count of the occurence of each predicted label,
  % which makes it more efficient then repeated calls to mode.
  minc = min(ys(:));
  maxc = max(ys(:));
  n = size(ys,1);
  cnt = zeros(n, maxc-minc+1);
  es  = zeros(size(ys));
  best = ones(n,1);
  for i=1:size(ys,2)
    yi = ys(:,i)-minc+1;
    cnt = cnt + sparse(1:n, yi, 1, n, maxc-minc+1);
    % does the value in this iteration beat the previous best? (for each instance)
    better = cnt((1:n)' + n*(best-1)) < cnt((1:n)' + n*(yi-1)); % column major indexing
    best(better) = yi(better);
    es(:,i) = best + minc-1;
  end
end

