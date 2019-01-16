function run_num_iterations_experiments(varargin)
  % Experiment to show the influence of the number of iterations on the results
  
  % Argument handling
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'verbose'), opts.verbose = true; end
  if ~isfield(opts,'quick'), opts.quick = false; end
  if ~isfield(opts,'use_cache'), opts.use_cache = true; end
  if ~isfield(opts,'cache_path'), opts.cache_path = '~/cache/domain-adaptation'; end
  if ~isfield(opts,'output_path'), opts.output_path = 'out/tables/num-iterations'; end
  if ~isfield(opts,'num_repeats'), opts.num_repeats = 200; end
  if ~isfield(opts,'num_iterations'), opts.num_iterations = [1,5:5:100]; end % parameter values to try

  if isfield(opts,'dataset')
    results = run_on(opts.dataset, opts);
    write_results(results, opts);
  else
    results = run_on(load_dataset('amazon'), opts);
    write_results(results, opts);
    results = run_on(load_dataset('office-caltech'), opts);
    write_results(results, opts);
    results = run_on(load_dataset('office-caltech','resnet50'), opts);
    write_results(results, opts);
    results = run_on(load_dataset('office-caltech','resnet50-no-augment'), opts);
    write_results(results, opts);
  end
end

function results = run_on(data, opts)
  results = struct();
  results.data = data;
  results.data.x = [];
  results.data.y = [];
  results.y_test    = cell(data.num_domain_pairs,1);
  results.ys        = cell(data.num_domain_pairs,1);
  results.mean_accs = cell(data.num_domain_pairs,1);
  results.mean_accs1 = cell(data.num_domain_pairs,1);
  results.svm_opts  = cell(data.num_domain_pairs,1);
  results.losses    = cell(data.num_domain_pairs,1);
  
  for src_tgt = 1:data.num_domain_pairs
    src = data.domain_pairs(src_tgt,1);
    tgt = data.domain_pairs(src_tgt,2);
    filename = sprintf('%s/num-iterations-%s-%s-%s-%s.mat', opts.cache_path, data.cache_filename, data.preprocessing, data.domains{src}, data.domains{tgt});
    
    if opts.use_cache && exist(filename,'file')
      if opts.verbose
        printf('%s %s->%s: cached\n', data.cache_filename, data.domains{src}(1), data.domains{tgt}(1));
      end
      clear('svm_opts','num_iterations','y_test','ys','yss','mean_accs','mean_accs1','losses')
      load(filename);
      if exist('yss','var') && ~exist('yss','mean_accs1')
        mean_accs1 = zeros(opts.num_repeats, numel(opts.num_iterations));
        for i=1:numel(opts.num_iterations)
          accs = bsxfun(@eq, yss{i}, y_test);
          mean_accs1(:,i) = mean(accs,1);
        end
      end
    elseif opts.quick
      if opts.verbose
        printf('%s %s->%s: skipped\n', data.cache_filename, data.domains{src}(1), data.domains{tgt}(1));
      end
      ys = [];
      y_test = [];
      svm_opts = struct();
      losses = [];
    else
      if opts.verbose
        printf('%s %s->%s: running\n', data.cache_filename, data.domains{src}(1), data.domains{tgt}(1));
      end
      x_train = data.x{src};
      x_test  = data.x{tgt};
      y_train = data.y{src};
      y_test  = data.y{tgt};
      
      [x_train_pp,x_test_pp] = preprocess(x_train, y_train, x_test, data.preprocessing);
      
      [~,svm_opts] = predict_liblinear_cv(x_train_pp,y_train,x_test_pp); % do CV only once
      
      num_iterations = opts.num_iterations;
      ys        = zeros(size(y_test,1), numel(opts.num_iterations));
      yss       = cell(numel(opts.num_iterations),1);
      mean_accs = zeros(size(y_test,1), numel(opts.num_iterations));
      mean_accs1 = zeros(opts.num_repeats, numel(opts.num_iterations));
      losses    = zeros(1, numel(opts.num_iterations));
      for i=1:numel(opts.num_iterations)
        if opts.verbose
          printf('%s %s->%s %d/%d   \r', data.name, data.domains{src}(1), data.domains{tgt}(1), i, numel(opts.num_iterations));
        end
        [y,ysi] = predict_adrem(x_train_pp,y_train,x_test_pp, 'num_repeats', opts.num_repeats, 'num_iterations',num_iterations(i), 'classifier',@predict_liblinear, 'classifier_opts',svm_opts);
        ys(:,i) = y;
        ysi = cell2mat(cellfun(@(x)x(:,end),ysi,'UniformOutput',false)');
        yss{i} = ysi;
        accs = bsxfun(@eq, ysi, y_test);
        mean_accs(:,i) = mean(accs,2); % mean over repeats
        mean_accs1(:,i) = mean(accs,1); % mean over samples
        losses(i) = tsvm_loss(x_train_pp, y_train, x_test_pp, y_test, [], svm_opts);
      end
      save(filename,'-v7','svm_opts','num_iterations','y_test','ys','yss','mean_accs','mean_accs1','losses');
    end
    results.num_iterations{src_tgt} = num_iterations;
    results.y_test{src_tgt} = y_test;
    results.ys{src_tgt} = ys;
    results.mean_accs{src_tgt} = mean_accs;
    results.mean_accs1{src_tgt} = mean_accs1;
    results.svm_opts{src_tgt} = svm_opts;
    results.losses{src_tgt} = losses;
  end
end

function write_results(results, opts)
  data = results.data;
  for src_tgt = 1:data.num_domain_pairs
    src = data.domain_pairs(src_tgt,1);
    tgt = data.domain_pairs(src_tgt,2);
    if isempty(results.ys{src_tgt}), continue; end;

    num_iterations = results.num_iterations{src_tgt};
    y_tgt = results.y_test{src_tgt};
    ys    = results.ys{src_tgt};
    acc   = bsxfun(@eq, y_tgt, ys);
    mean_acc = results.mean_accs{src_tgt};
    mean_acc1 = results.mean_accs1{src_tgt};
    losses = results.losses{src_tgt};
    
    filename = sprintf('%s/%s-%s-%s.dat', opts.output_path, data.cache_filename, data.domains{src}, data.domains{tgt});
    fprintf('%s\n',filename);
    f = fopen(filename,'wt');
    fprintf(f,'num_iterations');
    fprintf(f,'  ensemble_acc mean_acc std_acc');
    fprintf(f,'\n');
    for i=1:length(num_iterations)
      fprintf(f, '%d', num_iterations(i));
      fprintf(f, '  %f', mean(acc(:,i)));
      fprintf(f, '  %f', mean(mean_acc1(:,i)));
      fprintf(f, '  %f', std(mean_acc1(:,i)));
      fprintf(f, '\n');
    end
    fclose(f);
  end
  
  % Collect stats, only for common 'num_iterations'
  mean_acc = mean(results.mean_accs1{1},1);
  var_acc  = var(results.mean_accs1{1},0,1);
  std_acc  = std(results.mean_accs1{1},0,1);
  num_iterations = results.num_iterations{1};
  for src_tgt = 2:data.num_domain_pairs
    ma = mean(results.mean_accs1{src_tgt},1);
    va = var(results.mean_accs1{src_tgt},0,1);
    sa = std(results.mean_accs1{src_tgt},0,1);
    ni = results.num_iterations{src_tgt};
    [num_iterations,ia,ib] = intersect(num_iterations,ni);
    mean_acc = [mean_acc(:,ia); ma(:,ib)];
    var_acc  = [var_acc(:,ia); va(:,ib)];
    std_acc  = [std_acc(:,ia); sa(:,ib)];
  end
  size(mean_acc)
  var_acc = sum(var_acc) / data.num_domain_pairs.^2;
  
  filename = sprintf('%s/%s-avg.dat', opts.output_path, data.cache_filename);
  fprintf('%s\n',filename);
  f = fopen(filename,'wt');
  fprintf(f,'num_iterations');
  fprintf(f,'  mean_acc');
  fprintf(f,'  std_mean_acc');
  fprintf(f,'  std_acc');
  fprintf(f,'\n');
  %cell2mat(cellfun(@(ma)size(ma),results.mean_accs, 'UniformOutput',false))
  %mean_acc = cell2mat(cellfun(@(ma)mean(ma,1),results.mean_accs, 'UniformOutput',false));
  for i=1:length(num_iterations)
    fprintf(f, '%d', num_iterations(i));
    fprintf(f, '  %f', mean(mean_acc(:,i)));
    fprintf(f, '  %f', sqrt(var_acc(:,i)));
    fprintf(f, '  %f', mean(std_acc(:,i)));
    fprintf(f, '\n');
  end
  fclose(f);
end

