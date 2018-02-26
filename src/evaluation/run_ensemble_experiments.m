function run_ensemble_experiments(varargin)
  % Experiment to show the influence of the number of ensembled prediction on the results
  
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
  if ~isfield(opts,'output_path'), opts.output_path = 'out/tables/ensemble'; end
  if ~isfield(opts,'max_num_repeats'), opts.max_num_repeats = 1000; end
  if ~isfield(opts,'num_samples'), opts.num_samples = 2000; end % number of ensembles to sample
  if ~isfield(opts,'include_sampled'), opts.include_sampled = true; end
  if ~isfield(opts,'include_sampled_top'), opts.include_sampled_top = false; end
  if ~isfield(opts,'include_exact'), opts.include_exact = true; end
  
  results = run_on(load_dataset('amazon'), opts);
  write_results(results, opts);
  results = run_on(load_dataset('office-caltech'), opts);
  write_results(results, opts);
  results = run_on(load_dataset('office-caltech','resnet50'), opts);
  write_results(results, opts);
end

function results = run_on(data, opts)
  results = struct();
  results.data = data;
  results.data.x = [];
  results.data.y = [];
  results.y_test   = cell(data.num_domain_pairs,1);
  results.ys       = cell(data.num_domain_pairs,1);
  results.svm_opts = cell(data.num_domain_pairs,1);
  results.losses   = cell(data.num_domain_pairs,1);
  %results.ys = cell(1,data.num_domain_pairs);
  
  for src_tgt = 1:data.num_domain_pairs
    src = data.domain_pairs(src_tgt,1);
    tgt = data.domain_pairs(src_tgt,2);
    filename = sprintf('%s/ensemble-%s-%s-%s-%s.mat', opts.cache_path, data.cache_filename, data.preprocessing, data.domains{src}, data.domains{tgt});
    
    if opts.use_cache && exist(filename,'file')
      if opts.verbose
        printf('%s %s->%s: cached\n', data.name, data.domains{src}(1), data.domains{tgt}(1));
      end
      load(filename);
    elseif opts.quick
      if opts.verbose
        printf('%s %s->%s: skipped\n', data.name, data.domains{src}(1), data.domains{tgt}(1));
      end
      ys = [];
      y_test = [];
      svm_opts = struct();
      losses = [];
    else
      if opts.verbose
        printf('%s %s->%s: running\n', data.name, data.domains{src}(1), data.domains{tgt}(1));
      end
      x_train = data.x{src};
      x_test  = data.x{tgt};
      y_train = data.y{src};
      y_test  = data.y{tgt};
      
      [x_train_pp,x_test_pp] = preprocess(x_train, y_train, x_test, data.preprocessing);
      
      [~,svm_opts] = predict_liblinear_cv(x_train_pp,y_train,x_test_pp);
      
      ys = zeros(size(y_test,1), opts.max_num_repeats);
      losses = zeros(1, opts.max_num_repeats);
      %models = cell(1, opts.max_num_repeats);
      for i=1:opts.max_num_repeats
        if opts.verbose
          printf('%s %s->%s %d/%d   \r', data.name, data.domains{src}(1), data.domains{tgt}(1), i, opts.max_num_repeats);
        end
        [y] = predict_adrem(x_train_pp,y_train,x_test_pp, 'num_repeats',1, 'classifier',@predict_liblinear, 'classifier_opts',svm_opts);
        ys(:,i) = y;
        %models{i} = model{end};
        losses(i) = tsvm_loss(x_train_pp, y_train, x_test_pp, y_test, [], svm_opts);
      end
      save(filename,'-v7','svm_opts','y_test','ys', 'losses');
    end
    results.y_test{src_tgt} = y_test;
    results.ys{src_tgt} = ys;
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

    ensemble_sizes = 1:2:100;
    y_tgt  = results.y_test{src_tgt};
    ys     = results.ys{src_tgt};
    losses = results.losses{src_tgt};
    
    if opts.include_sampled
      filename = sprintf('%s/ensemble-sampled-%s-%s-%s-%s.mat', opts.cache_path, data.cache_filename, data.preprocessing, data.domains{src}, data.domains{tgt});
      if opts.use_cache && exist(filename,'file')
        load(filename);
      else
        [mean_acc, std_acc] = ensemble_accuracy(y_tgt, ys, ensemble_sizes, opts.num_samples);
        save(filename,'-v7','ensemble_sizes','mean_acc','std_acc');
      end
      mean_accs(src_tgt,:) = mean_acc;
      std_accs(src_tgt,:) = std_acc;
    end
    if opts.include_exact,       [mean_acc2, std_acc2] = ensemble_accuracy_exact(y_tgt, ys, ensemble_sizes); end
    if opts.include_sampled_top, [mean_acc3, std_acc3] = ensemble_accuracy(y_tgt, ys, ensemble_sizes, opts.num_samples, losses, 0.5); end
    
    
    filename = sprintf('%s/%s-%s-%s.dat', opts.output_path, data.cache_filename, data.domains{src}, data.domains{tgt});
    fprintf('%s\n',filename);
    f = fopen(filename,'wt');
    fprintf(f,'size');
    if opts.include_sampled,     fprintf(f,'  mean_acc_est std_acc_est'); end
    if opts.include_exact,       fprintf(f,'  mean_acc std_acc'); end
    if opts.include_sampled_top, fprintf(f,'  mean_acc_top std_acc_top'); end
    fprintf(f,'\n');
    for i=1:length(ensemble_sizes)
      fprintf(f, '%d', ensemble_sizes(i));
      if opts.include_sampled,     fprintf(f, '  %f %f', mean_acc(i), std_acc(i)); end
      if opts.include_exact,       fprintf(f, '  %f %f', mean_acc2(i), std_acc2(i)); end
      if opts.include_sampled_top, fprintf(f, '  %f %f', mean_acc3(i), std_acc3(i)); end
      fprintf(f, '\n');
    end
    fclose(f);
  end
  
  % Average over domains
  filename = sprintf('%s/%s-avg.dat', opts.output_path, data.cache_filename);
  fprintf('%s\n',filename);
  f = fopen(filename,'wt');
  fprintf(f,'size');
  if opts.include_sampled
    mean_acc = mean(mean_accs,1);
    std_acc = mean(std_accs,1);
    fprintf(f,'  mean_acc std_acc');
  end
  fprintf(f,'\n');
  for i=1:length(ensemble_sizes)
    fprintf(f, '%d', ensemble_sizes(i));
    if opts.include_sampled,     fprintf(f, '  %f %f', mean_acc(i), std_acc(i)); end
    fprintf(f, '\n');
  end
  fclose(f);
end

function [mean_acc, std_acc] = ensemble_accuracy(y_tgt, ys, ensemble_sizes, num_sample, losses, top_loss)
  % Estimate mean and variance of accuracy of different sizes majority-vote ensemble
  if nargin < 3
    ensemble_sizes = 1:2:100;
  end
  if nargin < 4
    num_sample = 100;
  end
  if nargin < 6
    top_loss = 1; % Keep only this fraction of the best losses
  end
  accs = zeros(num_sample, numel(ensemble_sizes));
  for i=1:numel(ensemble_sizes)
    sz = ensemble_sizes(i);
    for j=1:num_sample
      which = randperm(size(ys,2), sz);
      if top_loss < 1
        l = losses(which);
        [~,keep] = sort(l);
        which = which(keep(1:ceil(top_loss*i)));
      end
      y = mode(ys(:,which),2);
      accs(j,i) = mean(y == y_tgt);
      %losses(j,i) = tsvm_loss(x_src,y_src,x_tgt,y, [], svm_opts);
    end
  end
  
  mean_acc = mean(accs,1);
  std_acc  = std(accs, 0, 1);
end

function [mean_acc, std_acc] = ensemble_accuracy_exact(y_true, ys, ensemble_sizes)
  % For two classes, exactly calculating the expected majority-vote ensemble accuracy.
  % Assuming that the accuracy of each item is independent
  if nargin < 3
    ensemble_sizes = 1:3:500;
  end
  % probability of being correct for any sample
  p = mean(bsxfun(@eq, ys, y_true), 2);
  % we can see a majority vote as drawing from a binomial distribution, and asking if the total is ≥ 1/2*n
  
  for i=1:numel(ensemble_sizes)
    sz = ensemble_sizes(i);
    pi = 1-binocdf(idivide(sz,2), sz, p);
    if mod(sz,2)==0
      % ties count for 0.5
      pi = pi + 0.5 * binopdf(idivide(i,2), sz, p);
    end
    mean_acc(i) = mean(pi);
    std_acc(i)  = sqrt(sum(pi.*(1-pi)) / length(pi).^2);
  end
end

