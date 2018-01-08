function run_num_iterations_experiments(fast)
  if nargin < 1, fast = false; end
  oversamples = [1,0.2,5];
  for ios=1:numel(oversamples);
    os = oversamples(ios);
    method.filename = sprintf('icb-%g', os);
    method.method = @predict_abib;
    method.args = {'num_iterations',2000, 'oversample',os};
    
    preprocessing = 'joint-std';
    
    run_on(method, 'amazon', preprocessing, fast);
    run_on(method, 'office-caltech', preprocessing, fast);
    run_on(method, 'office-vgg-sumpool-fc6', preprocessing, fast);
  end
end

function run_on(method, data, preprocessing, fast)
  % load dataset
  out_path = 'out';
  cache_path = '~/cache/domain-adaptation';
  data = load_dataset(data);
  accs = [];
  accs_ensemble = [];
  mean_margin = [];
  mean_margin_ensemble = [];
  skipped = false;
  for src_tgt = 1:size(data.domain_pairs,1)
    src = data.domain_pairs(src_tgt,1);
    tgt = data.domain_pairs(src_tgt,2);
    % Cache?
    srep = '';
    filename = sprintf('%s/num-iter-%s-%s-%s-%s-%s%s.mat', cache_path, data.cache_filename, preprocessing, data.domains{src}, data.domains{tgt}, srep, method.filename);
    % prepare data
    printf('preparing %d %d         %s                              \r', src,tgt,filename);
    x_train = data.x{src};
    x_test  = data.x{tgt};
    y_train = data.y{src};
    y_test  = data.y{tgt};
    if exist(filename,'file')
      % Re-use cached value
      load(filename);
      if fast
        skipped = true;
        continue;
      end
    else
      printf('running %d %d                                       \n', src,tgt);
      [x_train,x_test] = preprocess(x_train, y_train, x_test, preprocessing);
      [y,ys] = method.method(x_train, y_train, x_test, method.args{:});
      save(filename,'-v7','ys');
    end
    % calculate accuracies
    acc = mean(ys == repmat(y_test,1,size(ys,2)), 1);
    if 1
      % fast calculation
      ys_ensemble = majority_votes(ys);
      acc_ensemble = mean(ys_ensemble == repmat(y_test,1,size(ys,2)), 1);
    else
      % slow calculation
      acc_ensemble = zeros(size(acc));
      for k=1:size(ys,2)
        y = mode(ys(:,1:k),2);
        acc_ensemble(k) = mean(y == y_test);
      end
    end
    accs(end+1,:) = acc;
    accs_ensemble(end+1,:) = acc_ensemble;
    % save
    % write to file
    outfile = sprintf('%s/plot-accuracy-%s-%s-%s-%s-%s', out_path, method.filename, data.filename, preprocessing, data.domains{src}, data.domains{tgt});
    write_result_plot([outfile '.tex'], acc);
    write_result_plot([outfile '-ensemble.tex'], acc_ensemble);
  end
  if ~skipped
    outfile = sprintf('%s/plot-accuracy-%s-%s-%s-avg', out_path, method.filename, data.filename, preprocessing);
    write_result_plot([outfile '.tex'], mean(accs,1));
    write_result_plot([outfile '-ensemble.tex'], mean(accs_ensemble,1));
  end
end

function write_result_plot(filename, accs, step)
  if nargin<3, step=5; end;
  fprintf('Writing %s     \r',filename);
  file = fopen(filename,'wt');
  fprintf(file,'\\addplot+[] coordinates{');
  for i=1:step:numel(accs)
    fprintf(file,'(%f,%f) ', i, 100*accs(i));
  end
  fprintf(file,'};\n');
  fclose(file);
end
