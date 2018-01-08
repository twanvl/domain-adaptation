function [y_tgt,ys_tgt] = predict_abib(x_src,y_src,x_tgt, varargin)
  % ABiB Domain adaptation.
  % Usage:
  %   y_tgt = predict_abib(x_src, y_src, x_tgt, options...)
  % 
  % Options can be given as a key,value pairs or as a struct.
  % Default settings are those used in the paper.
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'num_iterations'), opts.num_iterations = 500; end
  if ~isfield(opts,'alpha'),  opts.alpha = 0.5; end
  if ~isfield(opts,'classifier'), opts.classifier = @predict_liblinear_cv; end
  if ~isfield(opts,'classifier_opts'), opts.classifier_opts = struct(); end
  if ~isfield(opts,'classifier_opts_source'), opts.classifier_opts_source = opts.classifier_opts; end
  if ~isfield(opts,'use_source_C'), opts.use_source_C = false; end
  if ~isfield(opts,'use_target_C'), opts.use_target_C = true; end
  if ~isfield(opts,'bootstrap_source'), opts.bootstrap_source = false; end
  if ~isfield(opts,'bootstrap_target'), opts.bootstrap_target = true; end
  if ~isfield(opts,'balanced'), opts.balanced = true; end
  if ~isfield(opts,'oversample'), opts.oversample = 1; end
  if ~isfield(opts,'verbose'), opts.verbose = false; end
  if ~isfield(opts,'final'), opts.final = 'ensemble'; end
  if ~isfield(opts,'combine_models'), opts.combine_models = false; end
  
  % Classifier for the source
  [y_tgt,best_opts_src,model_src] = opts.classifier(x_src, y_src, x_tgt, opts.classifier_opts_source);
  y_tgt_from_src = y_tgt;
  if opts.use_source_C
    opts.classifier = @predict_liblinear;
    opts.classifier_opts = best_opts_src;
  elseif opts.use_target_C
    [~, best_opts_tgt] = opts.classifier(x_tgt,y_tgt,[], opts.classifier_opts);
    opts.classifier = @predict_liblinear;
    opts.classifier_opts = best_opts_tgt;
  end
  
  classes = unique(y_src);
  
  ys_tgt = zeros(size(x_tgt,1), opts.num_iterations);
  for it = 1:opts.num_iterations
    % New classifier for the source
    if opts.bootstrap_source
      n = size(x_src,1);
      if opts.balanced
        which = [];
        n = ceil(opts.oversample*n/numel(classes));
        for i=1:numel(classes)
          which_src = find(y_src == classes(i));
          which_src = which_src(randi(length(which_src),n,1));
          which = [which; which_src];
        end
      else
        which = ceil(rand(n,1)*n);
      end
      [y_tgt1,~,model_src] = opts.classifier(x_src(which,:), y_src(which,:), x_tgt, opts.classifier_opts_source);
    end
    
    % Classifier for the target
    if opts.bootstrap_target
      n = size(x_tgt,1);
      if opts.balanced
        which = [];
        n = ceil(opts.oversample*n/numel(classes));
        for i=1:numel(classes)
          which_tgt = find(y_tgt == classes(i));
          if ~isempty(which_tgt)
            which_tgt = which_tgt(randi(length(which_tgt),n,1));
            which = [which; which_tgt];
          end
        end
      else
        which = ceil(rand(n,1)*n);
      end
      [y_tgt2,~,model_tgt] = opts.classifier(x_tgt(which,:), y_tgt(which,:), x_tgt, opts.classifier_opts);
    else
      [y_tgt2,~,model_tgt] = opts.classifier(x_tgt, y_tgt, x_tgt, opts.classifier_opts);
    end
    
    % Combined classification
    model = combine_liblinear_models(opts.alpha,model_tgt, 1-opts.alpha,model_src);
    y_tgt = predict(y_tgt, sparse(x_tgt), model, '-q');
    ys_tgt(:,it) = y_tgt;
    
    if opts.verbose
      fprintf('%d \r',it);
    end
  end
  
  % final prediction
  if isequal(opts.final,'last')
    % use last y_tgt
  elseif isequal(opts.final,'ensemble')
    y_tgt = mode(ys_tgt, 2);
  else
    error('Unknown final aggregation method: %s', opts.final);
  end
end

