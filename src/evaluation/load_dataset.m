function data = load_dataset(data)
  % Load a dataset.
  %
  % Usage:
  %   data = load_dataset(name)
  %   data = load_dataset(struct('name',name, 'num_features',..))
  %
  % Arguments should either be a single name, or a struct:
  %   name          = name of the dataset
  %   num_features  = restrict dataset to this many features,
  %                   picks the first features!
  %   num_instances = restrict dataset to this many samples per domain
  %                   either a single value, or an array with one value per domain
  %                   picks random instances
  %
  % Returns a structure 'data' where
  %   data.name          = name of the dataset
  %   data.display_name  = nice looking name for use in tables and such
  %   data.filename      = unique name to use as a filename
  %   data.num_classes   = number of classes
  %   data.num_features  = number of features
  %   data.num_instances = number of instances (one value per domain)
  %   data.num_domains   = number of domains
  %   data.domains       = cell array of names of the domains
  %   data.x             = cell array of training data for each domain
  %   data.y             = cell array of labels for each domain
  % 
  % Experimental setting information:
  %   data.domain_pairs             = pairs of [source,target] domain ids to include
  %   data.num_repetitions          = number of repetitions to perform
  %   data.num_subsamples_per_class = number of instances per class to use from the source data (if set)
  %   data.preprocessing            = cell array of recommended preprocessing strategies to use
  % 
  % The cell arrays will have the same size.
  % The labels in y{i} will be from (0:num_classes-1)
  % Rows will be samples, columns the features
  
  % We assume that the data directory is located relative to the script
  datadir = 'data';
  
  if ~isstruct(data)
    if isequal(data,'amazon-small')
      data = struct('name','amazon', 'display_name','Amazon sentiment (small)', 'filename','amazon-small', 'num_features',5000, 'num_instances',2000);
    else
      data = struct('name',data, 'filename',data);
    end
  end
  
  % random seed for data shuffling
  rand('seed',1);
  
  if isequal(data.name, 'amazon')
    % Load the amazon sentiment dataset
    % This is prepared from files from http://www.cs.jhu.edu/~mdredze/datasets/sentiment/
    load([datadir, '/amazon.mat'], 'x','y','d');
    if ~isfield(data,'display_name')
      data.display_name = 'Amazon sentiment (ACL)';
    end
    data.domains = {'books', 'dvd', 'electronics', 'kitchen'};
    data.classes = [0,1];
    data.preprocessing = {'zscore','joint-std'};
    data.x = {};
    data.y = {};
    for i = 1:4
      data.y{i} = y(d==i,:);
      data.x{i} = x(d==i,:);
    end
  elseif isequal(data.name, 'amazon-subset')
    % Subset of the domains of the amazon data, as used in some papers
    data = load_dataset('amazon');
    data.name = 'amazon-subset';
    data.filename = 'amazon_subset';
    data.cache_filename = 'amazon';
    data.num_instances = 0;
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
  elseif isequal(data.name, 'amazon-reps')
    data = load_dataset('amazon');
    data.name = 'amazon-repeated';
    data.filename = 'amazon_repeated';
    data.num_repetitions = 10;
  elseif isequal(data.name, 'amazon-400')
    % Load the amazon dataset with 400 features,
    % from landmark paper; Gong, B.; Grauman, K.; and Sha, F.; 2013
    data.name = 'amazon-400';
    data.filename = 'amazon-400';
    data.display_name = 'Amazon 400';
    data.domains = {'books', 'dvd', 'elec', 'kitchen'};
    data.classes = [0,1];
    data.preprocessing = {'zscore','none','joint-std'};
    data.x = {};
    data.y = {};
    for i = 1:4
      load([datadir, '/amazon-400/',data.domains{i},'_400.mat'], 'fts', 'labels');
      data.x{i} = fts;
      data.y{i} = labels;
    end
  elseif isequal(data.name, 'amazon-400-standard')
    % Load the amazon dataset with 400 features,
    % from landmark paper; Gong, B.; Grauman, K.; and Sha, F.; 2013
    data.name = 'amazon-400-standard';
    data.filename = 'amazon-400-standard';
    data.display_name = 'Amazon 400 standard samples';
    data.domains = {'books', 'dvd', 'elec', 'kitchen'};
    data.classes = [0,1];
    data.preprocessing = {'zscore','none','joint-std'};
    data.num_repetitions = 20;
    data.x = {};
    data.y = {};
    data.train_idx = {};
    data.test_idx = {};
    for i = 1:4
      load([datadir, '/amazon-400/',data.domains{i},'_400.mat'], 'fts', 'labels');
      load([datadir, '/amazon-400/',data.domains{i},'_400_idx.mat'], 'train_idx', 'test_idx');
      data.x{i} = fts;
      data.y{i} = labels;
      data.train_idx{i} = train_idx;
      data.test_idx{i} = test_idx;
    end
  elseif isequal(data.name, 'amazon-400-subset')
    data = load_dataset('amazon-400');
    data.name = 'amazon-400-subset';
    data.filename = 'amazon-400-subset';
    data.cache_filename = 'amazon-400';
    data.num_instances = 0;
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
  elseif isequal(data.name, 'amazon-400-standard-subset')
    data = load_dataset('amazon-400-standard');
    data.name = 'amazon-400-standard-subset';
    data.filename = 'amazon-400-standard-subset';
    data.cache_filename = 'amazon-400-standard';
    data.num_instances = 0;
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
  
  elseif isequal(data.name, 'amazon-standard')
    % amazon, but using train_idx and test_idx from amazon-400-standard
    data.display_name = 'Amazon standard samples';
    data.domains = {'books', 'dvd', 'elec', 'kitchen'};
    data.classes = [0,1];
    data.preprocessing = {'zscore','joint-std'};
    data.num_repetitions = 20;
    data.x = {};
    data.y = {};
    data.train_idx = {};
    data.test_idx = {};
    load([datadir, '/amazon.mat'], 'x','y','d');
    for i = 1:4
      data.y{i} = y(d==i,:);
      data.x{i} = x(d==i,:);
      load([datadir, '/amazon-400/',data.domains{i},'_400_idx.mat'], 'train_idx', 'test_idx');
      data.train_idx{i} = train_idx;
      data.test_idx{i} = test_idx;
    end
  elseif isequal(data.name, 'amazon-standard-subset')
    data = load_dataset('amazon-standard');
    data.name = 'amazon-standard-subset';
    data.filename = 'amazon-standard-subset';
    data.cache_filename = 'amazon-standard';
    data.num_instances = 0;
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
    
  elseif isequal(data.name, 'office-caltech') || isequal(data.name, 'office_caltech_10_SURF')
    % Load the Office-Caltech datasets, with SURF features
    data.display_name = 'Office Caltech 10';
    data.filename = 'office_caltech_10_SURF_full';
    data.domains = {'amazon' 'Caltech10' 'dslr' 'webcam'};
    data.classes = 0:9;
    data.preprocessing = {'norm-row,zscore','joint-std'};
    data.x = {};
    data.y = {};
    for i=1:4
      load([datadir, '/office10/', data.domains{i}, '_SURF_L10.mat'], 'fts','labels');
      data.x{i} = fts;
      data.y{i} = labels - 1;
    end
  elseif isequal(data.name, 'office-caltech-standard')
    % The 'standard' experimental protocol on the office-caltech dataset:
    % 20 repetitions with 20 samples per domain (but 8 for DSLR)
    data = load_dataset('office-caltech');
    data.display_name = 'Office Caltech 10 (standard protocol)';
    data.filename = 'office_caltech_10_SURF_standard';
    data.cache_filename = 'office_caltech_10_SURF_standard';
    data.num_subsamples_per_class = [20,20,8,20];
    data.num_repetitions = 20;
  elseif isequal(data.name, 'office-caltech-reps')
    data = load_dataset('office-caltech');
    data.filename = 'office_caltech_10_SURF_full_repeated';
    data.cache_filename = 'office_caltech_10_SURF_full'; % use same cache filename
    data.num_repetitions = 10;
    
  elseif isequal(data.name, 'office-caltech-vgg-fc6')
    % Load the Office-Caltech datasets, with VGG features
    data = load_office_caltech_vgg(datadir,'fc6','');
  elseif isequal(data.name, 'office-caltech-vgg-fc7')
    data = load_office_caltech_vgg(datadir,'fc7','');
  elseif isequal(data.name, 'office-caltech-vgg-sumpool-fc6')
    data = load_office_caltech_vgg(datadir,'fc6','-sumpool');
  elseif isequal(data.name, 'office-caltech-vgg-sumpool-fc7')
    data = load_office_caltech_vgg(datadir,'fc7','-sumpool');
  elseif isequal(data.name, 'office-caltech-vgg-maxpool-fc6')
    data = load_office_caltech_vgg(datadir,'fc6','-maxpool');
  elseif isequal(data.name, 'office-caltech-vgg-maxpool-fc7')
    data = load_office_caltech_vgg(datadir,'fc7','-maxpool');
  elseif length(data.name) > 20 && isequal(data.name(1:14), 'office-caltech') && isequal(data.name(end-8:end),'-standard')
    data = load_dataset(data.name(1:end-9));
    data.filename = [data.filename, '-standard'];
    data.cache_filename = [data.cache_filename, '-standard'];
    data.display_name = [data.display_name, ' (standard protocol)'];
    data.num_subsamples_per_class = [20,20,8,20];
    data.num_repetitions = 20;
    
    
  elseif isequal(data.name, 'office-decaf7') || isequal(data.name, 'office_decaf7')
    % Load the Office datasets, with decaf 7 features
    data = load_office(datadir,'decaf7');
    data.display_name = 'Office Decaf 7';
  elseif isequal(data.name, 'office-decaf6') || isequal(data.name, 'office_decaf6')
    % Load the Office datasets, with decaf 6 features
    data = load_office(datadir,'decaf6');
    data.display_name = 'Office Decaf 6';
  elseif isequal(data.name, 'office-vgg-fc6')
    data = load_office_vgg(datadir,'fc6');
  elseif isequal(data.name, 'office-vgg-fc7')
    data = load_office_vgg(datadir,'fc7');
  elseif isequal(data.name, 'office-vgg-fc8')
    data = load_office_vgg(datadir,'fc8');
  elseif isequal(data.name, 'office-vgg-conv5')
    data = load_office_vgg(datadir,'conv5');
  elseif isequal(data.name, 'office-vgg-pool5')
    data = load_office_vgg(datadir,'pool5');
    for i=1:3
      data.x{i} = reshape(data.x{i},size(data.x{i},1),512*6*6);
    end
  elseif isequal(data.name, 'office-vgg-sumpool-fc6')
    data = load_office_vgg(datadir,'fc6','-sumpool');
  elseif isequal(data.name, 'office-vgg-sumpool-fc7')
    data = load_office_vgg(datadir,'fc7','-sumpool');
  elseif isequal(data.name, 'office-vgg-maxpool-fc6')
    data = load_office_vgg(datadir,'fc6','-maxpool');
  elseif isequal(data.name, 'office-vgg-maxpool-fc7')
    data = load_office_vgg(datadir,'fc7','-maxpool');
  elseif isequal(data.name, 'office-vgg-catpool-fc6')
    data = load_office_vgg(datadir,'fc6','-catpool');
    for i=1:3
      data.x{i} = reshape(permute(data.x{i},[2,1,3]), size(data.x{i},2),4096*10);
    end
  elseif isequal(data.name, 'office-vgg-catpool-fc7')
    data = load_office_vgg(datadir,'fc7','-catpool');
    for i=1:3
      data.x{i} = reshape(permute(data.x{i},[2,1,3]), size(data.x{i},2),4096*10);
    end
    
  elseif isequal(data.name, 'office-decaf7-standard')
    % Load the Office datasets, with decaf 7 features
    % standard protocol: 30 labeled training examples
    data = load_office(datadir,'decaf7');
    data.display_name = 'Office Decaf 7 (standard protocol)';
    data.filename = 'office_decaf7_standard';
    data.num_subsamples_per_class = [30,30,30];
    data.num_repetitions = 5;
    
  elseif isequal(data.name, 'cross-dataset-testbed')
    % Load the Cross-Dataset Testbed with decaf7 features (subsampled)
    data.display_name = 'Cross-Dataset Testbed (decaf7)';
    data.filename = 'cross-dataset-testbed-dense-decaf7';
    data.domains = {'caltech256','imagenet','sun'};
    data.classes = 1:40;
    data.preprocessing = {'truncate,joint-std','joint-std','zscore'};
    data.x = {};
    data.y = {};
    for i=1:3
      load([datadir, '/Cross-Dataset-Testbed_SubSampled/dense_', data.domains{i}, '_decaf7_subsampled2.mat'], 'fts', 'labels');
      data.x{i} = fts;
      data.y{i} = labels;
    end
  
  elseif isfield(data, 'x')
    % assume already loaded
  else
    error('Unknown dataset: %s', data.name)
  end
  
  if isfield(data, 'num_features') && data.num_features > 0 && data.num_features ~= size(data.x{1},2)
    printf('Selecting a subset of the features');
    for i = 1:numel(data.domains)
      data.x{i} = data.x{i}(:,1:data.num_features);
    end
  else
    data.num_features = size(data.x{1},2);
  end
  
  if isfield(data, 'num_instances') && any(data.num_instances > 0)
    %printf('Selecting a subset of the instances / permuting them');
    if numel(data.num_instances) == 1
      data.num_instances = repmat(data.num_instances, size(data.x));
    end
    for i = 1:numel(data.domains)
      which = randperm(length(data.y{i}), data.num_instances(i));
      %data.x{i} = data.x{i}(1:data.num_instances(i),:);
      %data.y{i} = data.y{i}(1:data.num_instances(i),:);
      data.x{i} = data.x{i}(which,:);
      data.y{i} = data.y{i}(which,:);
    end
  else
    data.num_instances = cellfun(@(x)size(x,1), data.x);
  end
  data.num_classes = numel(data.classes);
  data.num_domains = numel(data.domains);
  
  if ~isfield(data, 'num_repetitions')
    if isfield(data, 'train_idx')
      data.num_repetitions = size(data.train_idx,2);
    else
      data.num_repetitions = 1;
    end
  end
  
  if ~isfield(data, 'domain_pairs')
    data.domain_pairs = zeros(0,2);
    for src = 1:numel(data.domains)
      for tgt = 1:numel(data.domains)
        if src ~= tgt
          data.domain_pairs(end+1,:) = [src,tgt];
        end
      end
    end
  end
  data.num_domain_pairs = size(data.domain_pairs,1);
  
  if ~isfield(data,'cache_filename')
    data.cache_filename = data.filename;
  end
end

function data = load_office(datadir,decaf)
  data.name = sprintf('office_%s', decaf);
  data.display_name = sprintf('Office %s', decaf);
  data.filename = sprintf('office_%s',decaf);
  data.domains = {'amazon' 'dslr' 'webcam'};
  data.preprocessing = {'truncate,joint-std','zscore'};
  data.classes = 1:31;
  data.x = {};
  data.y = {};
  for i=1:3
    if 1
      % use prepared mat files
      load(sprintf('%s/office_%s/%s.mat',datadir,decaf,data.domains{i}),'x','y');
      data.x{i} = x;
      data.y{i} = y;
    else
      path = sprintf('%s/office_%s/%s_%s/',datadir,decaf,data.domains{i},decaf);
      files = dir([path,'*.mat']);
      for j=1:numel(files)
        load([path,files(j).name]);
      end
      data.x{i} = 0
    end
  end
end

function data = load_office_vgg(datadir,layer,prefix)
  if nargin<3
    prefix = '';
  end
  data.name = sprintf('office-vgg%s-%s', prefix,layer);
  data.display_name = sprintf('Office VGG%s %s', prefix,layer);
  data.filename = sprintf('office-vgg%s-%s',prefix,layer);
  data.domains = {'amazon' 'dslr' 'webcam'};
  data.preprocessing = {'truncate,joint-std','zscore'};
  data.classes = 1:31;
  data.x = {};
  data.y = {};
  for i=1:3
    load(sprintf('%s/office-vgg%s-%s-%s.mat',datadir,prefix,data.domains{i},layer),'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end

function data = load_office_caltech_vgg(datadir,layer,prefix)
  if nargin<3
    prefix = '';
  end
  data.name = sprintf('office-caltech-vgg%s-%s', prefix,layer);
  data.display_name = sprintf('Office Caltech 10 VGG%s %s', prefix,layer);
  data.filename = data.name;
  data.domains = {'amazon' 'Caltech' 'dslr' 'webcam'};
  data.classes = 1:10;
  data.preprocessing = {'truncate,joint-std','zscore'};
  data.x = {};
  data.y = {};
  for i=1:4
    load(sprintf('%s/office-caltech-vgg%s-%s-%s.mat', datadir, prefix, data.domains{i}, layer), 'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end
