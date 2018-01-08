function results = run_methods(data, methods, varargin)
  % Run all (or some) method on all (or some) datasets.
  % Store results in output files, and/or on the screen.
  %
  % Usage:
  %   run_methods(data, methods, [options])
  %   run_methods({data1,data2,..}, methods, [options])
  % 
  % Arguments:
  %   data    = (Name of) dataset to run experiments on, it will be loaded with load_dataset
  %   methods = A cell array of methods to run (default: all_methods())
  %   options = struct with extra options (or name,value pairs)
  % 
  % Options:
  %   'quick',false    Only use cached results
  %   'latex',true     Write latex tables for use in the paper (default: false)
  %
  % Returns results struct:
  %   results.data      = dataset
  %   results.methods   = which methods were used
  %   results.accs      = accuracy for (domain_pair, repetition, method)
  %   results.mean_accs = mean accuracy over repetitions (domain_pair, method)
  %   results.std_accs  = std deviation of accuracy
  %   results.mean_avg_accs = mean accuracy over repetitions (domain_pair, method)
  % 
  %   when called on multiple datasets, return a cell array of results
 
  
  % Argument handling
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'verbose'), opts.verbose = true; end
  if ~isfield(opts,'debug'), opts.debug = false; end
  if ~isfield(opts,'quick'), opts.quick = false; end
  if ~isfield(opts,'use_cache'), opts.use_cache = true; end
  if ~isfield(opts,'update_cache'), opts.update_cache = opts.use_cache; end
  if ~isfield(opts,'cache_path'), opts.cache_path = '~/cache/domain-adaptation'; end
  
  % Defaults
  if nargin < 2, methods = all_methods(); end
  
  if iscell(data)
    % Run on multiple datasets
    results = cellfun(@(d)run_methods(d, methods, opts), data, 'UniformOutput',false);
    return
  end
  
  % Load dataset?
  if ischar(data) || ~isfield(data,'x')
    data = load_dataset(data);
  end
  
  % Create output and cache directories
  if ~exist(opts.cache_path,'dir')
    mkdir(opts.cache_path);
  end
  
  % Print dataset info
  if opts.verbose
    plain_out = 1; % stdout
    fprintf(plain_out, '%s dataset, ', data.name);
    fprintf(plain_out, '%s features, ', data.features);
    fprintf(plain_out, '%s preprocessing\n', data.preprocessing);
    fprintf(plain_out, '%s========\n', repmat('=',1,length(data.name)));
    fprintf(plain_out, '%d features\n', data.num_features);
    num_instances = cellfun(@length, data.y);
    if min(num_instances) == max(num_instances)
      fprintf(plain_out, '%d instances\n', num_instances(1));
    else
      fprintf(plain_out, '%d-%d instances\n', min(num_instances), max(num_instances));
    end
    fprintf(plain_out, '%d domains\n', numel(data.domains));
    fprintf(plain_out, '%d classes\n', numel(data.classes));
    fprintf(plain_out, '\n');
  end
  
  % Find results from papers
  dummy_results = nan*zeros(numel(methods), data.num_domain_pairs);
  paper_results = results_from_papers(data);
  for i=1:numel(methods)
    if isfield(methods{i},'dummy')
      for j=1:numel(paper_results.methods)
        if isequal(methods{i}.name, paper_results.methods{j})
          dummy_results(i,:) = paper_results.accs(j,1:end-1);
          break;
        end
      end
    end
  end
  
  if opts.verbose
    % Table formating stuff
    domainWidth = max(6,max(cellfun(@length, data.domains))); % number of chars in names of domains
    methodNames = cellfun(@(m)m.name, methods, 'UniformOutput',false);
    methodWidths = max(6,cellfun(@length, methodNames)); % number of chars in names of methods
    srcTgtFormat = sprintf('%%-%ds   %%-%ds', domainWidth, domainWidth);
    methodFormat = arrayfun(@(x)sprintf('   %%-%ds', x), methodWidths, 'UniformOutput',false);
    resultFormat = arrayfun(@(x)sprintf('   %%-%d.3f', x), methodWidths, 'UniformOutput',false);
    lineWidth = domainWidth*2 + 3 + sum(methodWidths+3);
  
    fprintf(plain_out, 'Accuracy\n');
    fprintf(plain_out, srcTgtFormat, 'Source', 'Target');
    for i=1:numel(methods)
      fprintf(plain_out, methodFormat{i}, methodNames{i});
    end
    fprintf(plain_out, '\n');
    fprintf(plain_out, '%s\n', repmat('-',1,lineWidth));
  end
  
  % For all combinations of domains...
  accs = zeros(data.num_domain_pairs, data.num_repetitions, numel(methods));
  for src_tgt = 1:data.num_domain_pairs
    src = data.domain_pairs(src_tgt,1);
    tgt = data.domain_pairs(src_tgt,2);
    % Multiple repetitions?
    for rep = 1:data.num_repetitions
      if opts.verbose
        fprintf(plain_out, srcTgtFormat, data.domains{src}, data.domains{tgt});
      end
      
      % Get source and target datasets
      idx = [];
      if isfield(data,'num_subsamples_per_class')
        rand('seed',rep);
        y = data.y{src};
        for cls=1:numel(data.classes)
          idx_cls = find(y == data.classes(cls));
          keep = randperm(length(idx_cls), data.num_subsamples_per_class(src));
          idx = [idx, idx_cls(keep)];
        end
        x_train = data.x{src}(idx,:);
        y_train = data.y{src}(idx,:);
        x_test = data.x{tgt};
        y_test = data.y{tgt};
      elseif isfield(data,'train_idx')
        x_train = data.x{src}(data.train_idx{src}(:,rep),:);
        y_train = data.y{src}(data.train_idx{src}(:,rep),:);
        x_test = data.x{tgt}(data.test_idx{tgt}(:,rep),:);
        y_test = data.y{tgt}(data.test_idx{tgt}(:,rep),:);
        if opts.debug==2 && rep==3
          y_test(1:30)
          return
        end
      elseif src == tgt
        rand('seed',rep);
        train = rand(size(data.x{src},1),1) < 0.5;
        x_train = data.x{src}(train,:);
        x_test  = data.x{src}(~train,:);
        y_train = data.y{src}(train,:);
        y_test  = data.y{src}(~train,:);
      else
        x_train = data.x{src};
        x_test  = data.x{tgt};
        y_train = data.y{src};
        y_test  = data.y{tgt};
      end
      clear x_train_pp;
    
      % Run all methods
      for i_method = 1:numel(methods)
        method = methods{i_method};
        if isfield(method,'dummy')
          % Take results from another paper
          acc = dummy_results(i_method, src_tgt);
          
        else
          % Filename to use for caching results
          if data.num_repetitions > 1
            srep = sprintf('%d-',rep);
          else
            srep = '';
          end
          if ~isfield(method,'filename')
            method.filename = encode_parameters({method.method, method.args});
          end
          filename = sprintf('%s/%s-%s-%s-%s-%s%s.mat', opts.cache_path, data.cache_filename, data.preprocessing, data.domains{src}, data.domains{tgt}, srep, method.filename);
          if opts.debug
            fprintf('%s\n',filename);
          end
          
          if opts.use_cache && exist(filename,'file')
            % Re-use cached value
            load(filename,'y');
            
          elseif opts.quick
            % Skip
            y = [];
            
          else
            % Preprocess data
            if ~exist('x_train_pp','var')
              [x_train_pp,x_test_pp] = preprocess(x_train, y_train, x_test, data.preprocessing);
            end
            % Run method
            id = tic;
            cputime_before = cputime();
            if ~iscell(method.args)
              method.args = {method.args};
            end
            y = method.method(x_train_pp, y_train, x_test_pp, method.args{:});
            cputime_after = cputime();
            runtime = toc(id);
            runtime_cpu = cputime_after - cputime_before;
            if opts.update_cache && ~isempty(y)
              save(filename,'-v7','y','runtime', 'runtime_cpu', 'idx');
            end
          end
          
          % Calculate accuracy
          if isempty(y)
            acc = nan;
          else
            acc = mean(y == y_test);
          end
        end
        accs(src_tgt,rep,i_method) = acc;
        
        % Print
        if opts.verbose
          if data.num_repetitions > 1
            acc = mean(accs(src_tgt,1:rep,i_method));
          end
          fprintf(plain_out, resultFormat{i_method}, acc);
        end
      end
      if opts.verbose && data.num_repetitions > 1 && rep < data.num_repetitions
        fprintf(plain_out, '\r');
      end
    end
    if opts.verbose
      fprintf(plain_out, '\n');
    end
  end
  
  % Remove methods with nan results
  kept = [];
  for i = 1:numel(methods)
    if ~any(any(isnan(accs(:,:,i))))
      kept(end+1) = i;
    end
  end
  methods = methods(kept);
  accs = accs(:,:,kept);

  % add average over all domain pairs
  accs(end+1,:,:) = mean(accs,1);
  
  results = struct();
  results.data = data;
  results.data.x = [];
  results.data.y = [];
  results.methods = methods;
  results.accs    = accs;
  results.mean_accs = reshape(mean(accs,2),  [data.num_domain_pairs+1, numel(methods)]);
  results.std_accs  = reshape(std(accs,0,2), [data.num_domain_pairs+1, numel(methods)]);
  for i = 1:numel(methods)
    if isfield(methods{i},'dummy')
      results.std_accs(:,i) = nan; % we don't have stddevs for dummy methods
    end
  end
  results.mean_avg_accs = mean(results.mean_accs,1);
  results.std_avg_accs  = std(results.mean_accs,1);
  results.mean_std_accs = mean(results.std_accs,1);
  
  if opts.verbose
    % Output average over domains
    fprintf(plain_out, '%s\n', repmat('-',1,lineWidth));
    fprintf(plain_out, srcTgtFormat, 'avg','');
    for i_method = 1:numel(methods)
      accs_i = results.mean_accs(:,i_method);
      fprintf(plain_out, resultFormat{i_method}, mean(accs_i));
    end
    fprintf(plain_out, '\n');

    if data.num_repetitions > 1
      % Output average stddev over domains
      fprintf(plain_out, srcTgtFormat, 'avg std','');
      for i_method = 1:numel(methods)
        accs_i = results.std_accs(:,i_method);
        fprintf(plain_out, resultFormat{i_method}, mean(accs_i));
      end
      fprintf(plain_out, '\n');
    end
  end
end

