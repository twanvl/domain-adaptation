function [y_tgt,ys_tgt,models] = predict_adrem(x_src,y_src,x_tgt, varargin)
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'num_repeats'), opts.num_repeats = 11; end
  if ~isfield(opts,'num_iterations'), opts.num_iterations = 20; end
  if ~isfield(opts,'classifier'), opts.classifier = @predict_liblinear_cv; end
  if ~isfield(opts,'classifier_opts'), opts.classifier_opts = struct(); end
  if ~isfield(opts,'balanced'), opts.balanced = true; end
  if ~isfield(opts,'use_source_C'), opts.use_source_C = true; end
  if ~isfield(opts,'initial'), opts.initial = 0; end
  
  n_src = size(x_src,1);
  n_tgt = size(x_tgt,1);
  classes = unique(y_src);
  if ~isfield(opts,'class_balance'), opts.class_balance = ones(1,numel(classes)); end
  
  % Classifier for the source
  [y_tgt_src,best_opts_src,model_src] = opts.classifier(x_src, y_src, x_tgt, opts.classifier_opts);
  if opts.use_source_C
    opts.classifier = @predict_liblinear;
    opts.classifier_opts = best_opts_src;
  end

  ys_reps = zeros(n_tgt, opts.num_repeats);
  if nargout >= 2
    ys_tgt = cell(opts.num_repeats, 1);
  end
  if nargout >= 3
    models = cell(opts.num_repeats, opts.num_iterations+1);
  end
  for rep = 1:opts.num_repeats
    % Start from source classification
    y_tgt = y_tgt_src;
    if nargout >= 2
      ys_tgt{rep} = zeros(n_tgt, opts.num_iterations+1);
      ys_tgt{rep}(:,1) = y_tgt_src;
    end
    if nargout >= 3
      models{rep,1} = model_src;
    end
    
    for it = 1:opts.num_iterations
      n_tgt_sample = n_tgt * (opts.initial + (1-opts.initial)*(it / opts.num_iterations));
      
      if opts.balanced
        which_tgt = [];
        for i=1:numel(classes)
          % We have to take n_i samples out of length(which_i)
          n_i = ceil(opts.class_balance(i)*n_tgt_sample/numel(classes));
          which_i = find(y_tgt == classes(i));
          if length(which_i) == 0
            continue; % we completely lost a class :(
          end
          % sample once without replacement
          % the rest: copy all samples
          copies = floor(n_i / length(which_i));
          idx = [repmat(1:length(which_i), 1, copies), randperm(length(which_i), n_i-copies*length(which_i))];
          which_tgt = [which_tgt; which_i(idx(:))];
        end
      else
        which_tgt = randperm(n_tgt, ceil(n_tgt_sample));
      end
      
      x_train = [x_src; x_tgt(which_tgt,:)];
      y_train = [y_src; y_tgt(which_tgt,:)];
      
      [y_tgt,~,model] = opts.classifier(x_train, y_train, x_tgt, opts.classifier_opts);
      
      if nargout >= 2
        ys_tgt{rep}(:,it+1) = y_tgt;
      end
      if nargout >= 3
        models{rep,it+1} = model;
      end
    end
    ys_reps(:,rep) = y_tgt;
  end
  
  % final prediction
  y_tgt = mode(ys_reps, 2);
end


