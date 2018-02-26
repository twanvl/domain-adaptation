function run_all_and_write_tables(varargin)
  % Script to run all experiments used in the paper
  
  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'quick'), opts.quick = true; end
  if ~isfield(opts,'verbose'), opts.verbose = false; end
  if ~isfield(opts,'output_path'), opts.output_path = 'out/tables'; end
  
  if opts.quick
    fprintf('Using cached results only, pass (''quick'',0) to actually run\n');
  end
  
  run_and_write_tables('amazon-standard-subset', opts);
  run_and_write_tables('amazon-repeated', opts);
  run_and_write_tables('office-caltech-standard', opts);
  run_and_write_tables('office-caltech-repeated', opts);
  run_and_write_tables('office-caltech', opts);
  run_and_write_tables('office', opts);
  run_and_write_tables('cross-dataset-testbed', opts);
end

function run_and_write_tables(dataset, opts)
  methods = all_methods;
  baselines = methods(1:2);
  our_methods = methods(3:end);
  baselines{end}.space_after = true;
  our_methods{1}.space_before = true;
  
  if isequal(dataset,'amazon-standard-subset') || isequal(dataset,'amazon-repeated')
    % Amazon
    data = cellfun(@(x)load_dataset(dataset,x), {'400','full'}, 'UniformOutput',false);
    methods = [baselines, all_methods_literature('toolbox_sa',true), our_methods];
    results = run_methods(data, methods, opts);
    write_table(sprintf('%s/%s.tex',opts.output_path,dataset), results);
    
  elseif isequal(dataset,'office-caltech') || isequal(dataset,'office-caltech-standard') || isequal(dataset,'office-caltech-repeated')
    % Office Caltech (standard)
    data = cellfun(@(x)load_dataset(dataset,x), {'surf','resnet50'}, 'UniformOutput',false);
    methods = [baselines, dummy_methods('SA'), all_methods_literature('include_gfk',false, 'include_tca',false), our_methods];
    results = run_methods(data, methods, opts);
    write_table(sprintf('%s/%s.tex',opts.output_path,dataset), results);
    
  elseif isequal(dataset,'office')
    % Office 31 (standard)
    data = cellfun(@(x)load_dataset(dataset,x), {'decaf','resnet50','raw-resnet'}, 'UniformOutput',false);
    %data{end}.display_name = 'Deep Neural Networks';
    data{end}.display_name = 'Deep Neural Networks (based on ResNet)';
    %dummys = dummy_methods('SA','CORAL','DLID','DDC','DAN','DANN','BP','RTN','TDS');
    dummys1 = dummy_methods('SA');
    dummys2 = dummy_methods('CORAL','DDC','DAN','RTN','RevGrad','JAN-A');
    methods = [baselines, dummys1, all_methods_literature(), dummys2, our_methods];
    results = run_methods(data, methods, opts);
    write_table(sprintf('%s/%s.tex',opts.output_path,dataset), results);
    
  elseif isequal(dataset,'cross-dataset-testbed')
    % Cross Dataset Testbed
    data = {};
    data{1} = load_dataset(dataset, 'decaf-fc7');
    data{2} = load_dataset(dataset, 'decaf-fc7','truncate,joint-std');
    data{1}.display_name = 'DECAF-fc7 features';
    data{2}.display_name = 'DECAF-fc7 features, rectified';
    dummys1 = dummy_methods('SA');
    dummys2 = dummy_methods('CORAL');
    methods = [baselines, dummys1, all_methods_literature(), dummys2, our_methods];
    results = run_methods(data, methods, opts);
    write_table(sprintf('%s/%s.tex',opts.output_path,dataset), results);
  end
end

function write_table(filename, results, varargin)
  data = results{1}.data;

  if length(varargin) == 1 && isstruct(varargin{1})
    opts = varargin{1};
  else
    opts = struct(varargin{:});
  end
  if ~isfield(opts,'include_avg') opts.include_avg = true; end;
  if ~isfield(opts,'what') opts.what = 'mean'; end;
  opts.include_std = isequal(opts.what, 'mean') && data.num_repetitions > 1;
  opts.inline_std = opts.include_std && data.num_domain_pairs <= 6;
  opts.include_avg_std = 1;
  opts.subheaders = numel(results) > 1;
  num_columns = data.num_domain_pairs+opts.include_avg;
  
  fprintf('%s\n',filename);
  latex_out = fopen(filename,'wt');
  
  fprintf(latex_out, '%% %s\n', data.name);
  fprintf(latex_out, '%% mean over %d runs\n', data.num_repetitions);
  fprintf(latex_out, '\\begin{tabular}{%sll%s}\n', '@{\hspace*{\leftsep}}', repmat('@{\hspace*{\colsep}}c',1, num_columns));
  
  fprintf(latex_out, '\\hlinetop\n');
  fprintf(latex_out, '&');
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

  % Best result for each column
  scores = cellfun(@(res)res.mean_accs,results, 'UniformOutput',false);
  scores = cat(2, scores{:});
  bests = max(scores,[],2);
  
  % Print table
  for i_variant = 1:numel(results)
    res = results{i_variant};
    if opts.subheaders
      if i_variant > 1
        fprintf(latex_out, '\\hlinemid\n');
      end
      fprintf(latex_out, '%%--------------------------------------------------\n');
      fprintf(latex_out, '\\multicolumn{%d}{l}{%s}\\\\[\\subheadersep]\n', num_columns, res.data.display_name);
    end
    for i_method = 1:numel(res.methods)
      method = res.methods{i_method};
      fprintf(latex_out, '%% %s, %s, %s\n', method.name, res.data.features, res.data.preprocessing);
      fprintf(latex_out, '& %s', method.name);
      for src_tgt = 1:num_columns
        if isequal(opts.what, 'mean')
          score = res.mean_accs(src_tgt,i_method);
          score_std = res.std_accs(src_tgt,i_method);
          best = bests(src_tgt);
        elseif isequal(opts.what, 'std')
          score = res.std_accs(src_tgt,i_method);
        end
        if isnan(score)
          fprintf(latex_out, ' & -');
        else
          if score >= best
            tag = '\best';
          %elseif src_tgt
            % TODO: significance test
          else
            tag = '';
          end
          fprintf(latex_out, ' & %s{%.1f}', tag, 100*score);
          if opts.inline_std && ~isnan(score_std) && (src_tgt ~= data.num_domain_pairs+1 || opts.include_avg_std)
            fprintf(latex_out, '\\inlinestd{%.1f}', 100*score_std);
          end
        end
      end
      if opts.include_std && ~opts.inline_std && ~all(isnan(res.std_accs(:,i_method)))
        % standard deviation on next line
        fprintf(latex_out, '\\\\\n');
        fprintf(latex_out, '\\stddevs&');
        for src_tgt = 1:num_columns
          score_std = res.std_accs(src_tgt,i_method);
          fprintf(latex_out, '&\\withstd{%.1f}', 100*score_std);
        end
      end
      fprintf(latex_out, '\\\\');
      if i_method+1 <= numel(res.methods) && (isfield(method,'space_after') || isfield(res.methods{i_method+1},'space_before'))
        fprintf(latex_out, '[\\methodsep]');
      end
      fprintf(latex_out, '\n');
    end
  end
  fprintf(latex_out, '\\hlinebot\n');
  fprintf(latex_out, '\\end{tabular}\n');
  fclose(latex_out);
end

function [better,p] = sign_test(y1, y2, alpha, scale)
  % Test the null hypothesis that y1>y2 and y1<y2 are equally likely
  % using a one sided binomial (exact) sign test.
  %
  % The alternative is that y2 is better.
  %
  % Multiply counts by scale if the y are not independent.
  % 
  % Returns true if null hypothesis is rejected (y2 is significantly better)
  if nargin < 4
    scale = 1;
  end
  n1 = nnz(y1 > y2);
  n2 = nnz(y1 < y2);
  p = binocdf(n1,n1+n2,0.5);
  better = p < alpha;
end
