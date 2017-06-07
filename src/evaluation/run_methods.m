function run_methods(datasets, methods, varargin)
  % Run all (or some) method on all (or some) datasets.
  % Store results in output files, and/or on the screen.
  %
  % Usage:
  %   run_methods(datasets, methods, [options])
  % 
  % Options:
  %   'latex',true     Write latex tables for use in the paper (default: false)
  
  % Argument handling
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'self'), opts.self = false; end
  if ~isfield(opts,'quick'), opts.quick = false; end
  if ~isfield(opts,'latex'), opts.latex = false; end
  if ~isfield(opts,'latex_plot'), opts.latex_plot = false; end
  if ~isfield(opts,'include_avg'), opts.include_avg = true; end
  if ~isfield(opts,'output_path'), opts.output_path = 'out'; end
  if ~isfield(opts,'use_cache'), opts.use_cache = true; end
  if ~isfield(opts,'update_cache'), opts.update_cache = opts.use_cache; end
  if ~isfield(opts,'cache_path'), opts.cache_path = '~/cache/domain-adaptation'; end
  if ~isfield(opts,'preprocessing'), opts.preprocessing = []; end
  
  % Defaults
  if nargin < 1, datasets = all_datasets(); end
  if nargin < 2, methods  = all_methods(); end
  
  if ischar(datasets) || isstruct(datasets)
    datasets = {datasets}; % allow  run_methods(single_dataset)
  end
  
  % Open output files
  plain_out = 1; % stdout
  
  % Create output and cache directories
  if ~exist(opts.output_path,'dir')
    mkdir(opts.output_path);
  end
  if ~exist(opts.cache_path,'dir')
    mkdir(opts.cache_path);
  end
  
  % All methods need a filename for caching
  for i = 1:numel(methods)
    if ~isfield(methods{i},'filename')
      methods{i}.filename = encode_parameters({methods{i}.method,methods{i}.args});
    end
  end
  
  % For each dataset...
  for i_data = 1:numel(datasets)
    data = datasets{i_data};
    if ischar(data) || ~isfield(data,'x')
      data = load_dataset(data);
    end
    
    % Preprocessing to use
    if ~isempty(opts.preprocessing)
      preprocessing = opts.preprocessing;
    else
      preprocessing = 'preferred';
    end
    
    % Dataset info
    fprintf(plain_out, '%s dataset\n', data.display_name);
    fprintf(plain_out, '%s========\n', repmat('=',1,length(data.name)));
    fprintf(plain_out, '%d features\n', data.num_features);
    if min(data.num_instances) == max(data.num_instances)
      fprintf(plain_out, '%d instances\n', data.num_instances(1));
    else
      fprintf(plain_out, '%d-%d instances\n', min(data.num_instances), max(data.num_instances));
    end
    fprintf(plain_out, '%d domains\n', data.num_domains);
    fprintf(plain_out, '%d classes\n', data.num_classes);
    fprintf(plain_out, '%s preprocessing\n', preprocessing);
    fprintf(plain_out, '\n');
    
    % Methods to use
    % usually these are all methods, except for some with predefined results that we might not have
    methods_for_dataset = {};
    for i=1:numel(methods)
      if isfield(methods{i},'results')
        if ~isequal(preprocessing, 'preferred')
          continue;
        elseif ~isfield(methods{i}.results, data.filename)
          continue; % Don't have predefined results for this method
        end
      end
      methods_for_dataset{end+1} = methods{i};
    end
    
    % Table formating stuff
    domainWidth = max(6,max(cellfun(@length, data.domains))); % number of chars in names of domains
    methodNames = cellfun(@(m)m.name, methods_for_dataset, 'UniformOutput',false);
    methodWidths = max(6,cellfun(@length, methodNames)); % number of chars in names of methods
    srcTgtFormat = sprintf('%%-%ds   %%-%ds', domainWidth, domainWidth);
    methodFormat = arrayfun(@(x)sprintf('   %%-%ds', x), methodWidths, 'UniformOutput',false);
    resultFormat = arrayfun(@(x)sprintf('   %%-%d.3f', x), methodWidths, 'UniformOutput',false);
    lineWidth = domainWidth*2 + 3 + sum(methodWidths+3);
    
    %for i_pp = 1:numel(preprocessings)
    
    fprintf(plain_out, 'Accuracy\n');
    fprintf(plain_out, srcTgtFormat, 'Source', 'Target');
    for i=1:numel(methods_for_dataset)
      fprintf(plain_out, methodFormat{i}, methodNames{i});
    end
    fprintf(plain_out, '\n');
    if isequal(preprocessing,'preferred')
      fprintf(plain_out, srcTgtFormat, '', '');
      for i = 1:numel(methods_for_dataset)
        pp = preprocessing_for_method(data, methods_for_dataset{i}, preprocessing);
        if length(pp) > methodWidths(i), pp = pp(1:methodWidths(i)); end;
        fprintf(plain_out, methodFormat{i}, pp);
      end
      fprintf(plain_out, '\n');
    end
    fprintf(plain_out, '%s\n', repmat('-',1,lineWidth));
    
    %
    if opts.self
      data.domain_pairs = repmat((1:numel(data.domains))',1,2);
      data.num_domain_pairs = numel(data.domains);
    end
    
    % For all combinations of domains...
    accs = zeros(data.num_domain_pairs, data.num_repetitions, numel(methods_for_dataset));
    for src_tgt = 1:data.num_domain_pairs
      src = data.domain_pairs(src_tgt,1);
      tgt = data.domain_pairs(src_tgt,2);
      % Multiple repetitions?
      for rep = 1:data.num_repetitions
        % Get source and target datasets
        if src == tgt
          train = rand(size(data.x{src},1),1) < 0.5;
          x_train = data.x{src}(train,:);
          x_test  = data.x{src}(~train,:);
          y_train = data.y{src}(train,:);
          y_test  = data.y{src}(~train,:);
        elseif isfield(data,'num_g_per_class')
          rand('seed',rep);
          y = data.y{src};
          idx = [];
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
        else
          x_train = data.x{src};
          x_test  = data.x{tgt};
          y_train = data.y{src};
          y_test  = data.y{tgt};
        end
        fprintf(plain_out, srcTgtFormat, data.domains{src}, data.domains{tgt});
        
        for i_method = 1:numel(methods_for_dataset)
          method = methods_for_dataset{i_method};
          % Preprocessing to use
          if isfield(method,'results')
            % This method has predefined results for this dataset
            results = getfield(method.results, data.filename);
            acc = results(src_tgt);
          else
            pp = preprocessing_for_method(data, method, preprocessing);
            % Filename to use for caching results
            if data.num_repetitions > 1
              srep = sprintf('%d-',rep);
            else
              srep = '';
            end
            filename = sprintf('%s/%s-%s-%s-%s-%s%s.mat', opts.cache_path, data.cache_filename, pp, data.domains{src}, data.domains{tgt}, srep, method.filename);
            if opts.use_cache && exist(filename,'file')
              % Re-use cached value
              load(filename,'y');
            elseif opts.quick
              y = [];
            else
              % Run method
              [x_train_pp,x_test_pp] = preprocess(x_train, y_train, x_test, pp);
              id = tic;
              cputime_before = cputime();
              y = method.method(x_train_pp, y_train, x_test_pp, method.args{:});
              cputime_after = cputime();
              runtime = toc(id);
              runtime_cpu = cputime_after - cputime_before;
              if opts.update_cache && ~isempty(y)
                save(filename,'-v7','y','runtime', 'runtime_cpu');
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
          if data.num_repetitions > 1
            acc = mean(accs(src_tgt,1:rep,i_method));
          end
          fprintf(plain_out, resultFormat{i_method}, acc);
        end
        if data.num_repetitions > 1 && rep < data.num_repetitions
          fprintf(plain_out, '\r');
        end
      end
      fprintf(plain_out, '\n');
    end
    mean_accs = reshape(mean(accs,2), [data.num_domain_pairs, numel(methods_for_dataset)]);
    std_accs  = reshape(std(accs,0,2), [data.num_domain_pairs, numel(methods_for_dataset)]);
    
    % Output average over domains
    fprintf(plain_out, '%s\n', repmat('-',1,lineWidth));
    fprintf(plain_out, srcTgtFormat, 'avg','');
    for i_method = 1:numel(methods_for_dataset)
      accs_i = mean_accs(:,i_method);
      fprintf(plain_out, resultFormat{i_method}, mean(accs_i));
    end
    fprintf(plain_out, '\n');

    if data.num_repetitions > 1
      % Output average stddev over domains
      fprintf(plain_out, srcTgtFormat, 'avg std','');
      for i_method = 1:numel(methods_for_dataset)
        accs_i = std_accs(:,i_method);
        fprintf(plain_out, resultFormat{i_method}, mean(accs_i));
      end
      fprintf(plain_out, '\n');
    end
    
    % calculate average accuracy
    mean_avg_accs = mean(mean_accs,1);
    std_avg_accs = std(mean_accs,1);
    mean_std_accs = mean(std_accs,1);
    
    % Output to latex
    if isequal(preprocessing,'preferred')
      pp = '';
    else
      pp = ['_',preprocessing];
    end
    if opts.latex
      filename = sprintf('%s/table_accuracy_%s%s.tex', opts.output_path, data.filename, pp);
      latex_out = fopen(filename,'wt');
      fprintf(latex_out, '%% mean over %d runs\n', data.num_repetitions);
      fprintf(latex_out, '\\begin{tabular}{l%s}\n', repmat('@{\hspace*{\colsep}}c',1,data.num_domain_pairs+opts.include_avg));
      fprintf(latex_out, '\\hlinetop\n');
      for src_tgt = 1:data.num_domain_pairs
        src = data.domain_pairs(src_tgt,1);
        tgt = data.domain_pairs(src_tgt,2);
        fprintf(latex_out, ' & %s$\\to$%s', toupper(data.domains{src}(1)), toupper(data.domains{tgt}(1)));
      end
      if opts.include_avg
        fprintf(latex_out, ' & avg');
      end
      fprintf(latex_out, '\\\\\n');
      fprintf(latex_out, '\\hlinemid\n');
      for i_method = 1:numel(methods_for_dataset)
        method = methods_for_dataset{i_method};
        fprintf(latex_out, '%s', methodNames{i_method});
        for src_tgt = 1:data.num_domain_pairs
          if all(mean_accs(src_tgt,i_method) >= mean_accs(src_tgt,:))
            isbest = '\best';
          else
            isbest = '';
          end
          fprintf(latex_out, ' & %s{%.1f}', isbest, 100*mean_accs(src_tgt,i_method));
        end
        if opts.include_avg
          if all(mean_avg_accs(i_method) >= mean_avg_accs)
            isbest = '\best';
          else
            isbest = '';
          end
          fprintf(latex_out, ' & %s{%.1f}', isbest, 100*mean_avg_accs(i_method));
        end
        fprintf(latex_out, '\\\\\n');
      end
      fprintf(latex_out, '\\hlinebot\n');
      fprintf(latex_out, '\\end{tabular}\n');
      fclose(latex_out);
    end
    
    if opts.latex && data.num_repetitions > 1
      filename = sprintf('%s/table_accuracy_std_%s%s.tex', opts.output_path, data.filename, pp);
      latex_out = fopen(filename,'wt');
      fprintf(latex_out, '%% stddev over %d runs\n', data.num_repetitions);
      fprintf(latex_out, '\\begin{tabular}{l%s}\n', repmat('@{\hspace*{\colsep}}c',1,data.num_domain_pairs+opts.include_avg));
      fprintf(latex_out, '\\hlinetop\n');
      for src_tgt = 1:data.num_domain_pairs
        src = data.domain_pairs(src_tgt,1);
        tgt = data.domain_pairs(src_tgt,2);
        fprintf(latex_out, ' & %s$\\to$%s', toupper(data.domains{src}(1)), toupper(data.domains{tgt}(1)));
      end
      if opts.include_avg
        fprintf(latex_out, ' & avg');
      end
      fprintf(latex_out, '\\\\\n');
      fprintf(latex_out, '\\hlinemid\n');
      for i_method = 1:numel(methods_for_dataset)
        method = methods_for_dataset{i_method};
        if isfield(method,'results')
          continue; % no std dev for results from papers
        end
        fprintf(latex_out, '%s', methodNames{i_method});
        for src_tgt = 1:data.num_domain_pairs
          fprintf(latex_out, ' & {%.1f}', 100*std_accs(src_tgt,i_method));
        end
        if opts.include_avg
          fprintf(latex_out, ' & {%.1f}', 100*mean_std_accs(i_method));
        end
        fprintf(latex_out, '\\\\\n\n');
      end
      fprintf(latex_out, '\\hlinebot\n');
      fprintf(latex_out, '\\end{tabular}\n');
      fclose(latex_out);
    end
    
    if opts.latex_plot && isfield(methods{1},'key')
      filename = sprintf('%s/plot_%s_%s%s.tex', opts.output_path, methods{1}.key_name, data.filename, pp);
      latex_out = fopen(filename,'wt');
      fprintf(latex_out,'\\addplot+[] coordinates{\n');
      for i_method = 1:numel(methods_for_dataset)
        method = methods_for_dataset{i_method};
        fprintf(latex_out, '  (%f,%f) +- (%f,%f)\n', method.key, 100*mean_avg_accs(i_method), 0,100*std_avg_accs(i_method));
      end
      fprintf(latex_out,'};\n');
      fclose(latex_out);
    end
  end
end

function preprocessing = preprocessing_for_method(data, method, preprocessing)
  if isequal(preprocessing,'preferred')
    preprocessing = data.preprocessing{1};
    if isfield(method, 'results')
      preprocessing = 'predefined';
    elseif isfield(method, 'preferred_preprocessing')
      % Find the first preprocessing that makes sense for this data that is also liked by the method
      for i=1:numel(data.preprocessing)
        if ismember(data.preprocessing{i}, method.preferred_preprocessing)
          preprocessing = data.preprocessing{i};
          break;
        end
      end
    end
  else
    preprocessing = preprocessing;
  end
end
