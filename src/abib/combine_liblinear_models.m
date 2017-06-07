function model = combine_liblinear_models(a, m1, b, m2)
  % Make a linear combination of two liblinear models
  % 
  % Usage:
  %   model = combine_liblinear_models(a, m1, b, m2)
  % 
  % Predictions will be  a*m1+b*m2
  
  if ~isequal(m1.Parameters,m2.Parameters)
    error('Model parameters differ');
  end
  if ~isequal(m1.bias,m2.bias)
    error('Model bias differs');
  end
  if ~isequal(m1.nr_feature,m2.nr_feature)
    error('Model nr of features differ');
  end
  
  labels = union(m1.Label, m2.Label);
  % note: order matters
  model = struct();
  model.Parameters = m1.Parameters;
  model.nr_class = numel(labels);
  model.nr_feature = m1.nr_feature;
  model.bias = m1.bias;
  model.Label = labels;
  
  % Is this a two class problem?
  if model.nr_class == 2 && size(m1.w,1) == 1 && size(m2.w,1) == 1
    % two class problem, single decision boundary
    model.w = zeros(1, size(m1.w,2));
  else
    % one vs all
    model.w = zeros(model.nr_class, size(m1.w,2));
  end
  
  % We can't just combine w, since labels might be in a different order
  % plus, some classes might be missing from one of the classifiers
  for i=1:size(model.w,1)
    idx1 = find(m1.Label==model.Label(i),1);
    if isempty(idx1)
      w1 = zeros(1,size(m1.w,2));
    elseif idx1 > size(m1.w,1)
      w1 = -mean(m1.w,1);
    else
      w1 = m1.w(idx1,:);
    end
    idx2 = find(m2.Label==model.Label(i),1);
    if isempty(idx2)
      w2 = zeros(1,size(m2.w,2));
    elseif idx2 > size(m2.w,1)
      w2 = -mean(m2.w,1);
    else
      w2 = m2.w(idx2,:);
    end
    model.w(i,:) = a*w1 + b*w2;
  end
end

