function data = load_dataset(data, features, preprocessing)
  % Load a dataset.
  %
  % Usage:
  %   data = load_dataset(name)
  %   data = load_dataset(name, [features], [preprocessing])
  %   data = load_dataset(struct('name',name, 'features',features, ..))
  %
  % Arguments should either be a single name, or a struct:
  %   name          = name of the dataset (+protocol)
  %   features      = which set of features to use
  %   preprocessing = preprocessing strategy to use (see preprocess.m)
  %
  % Returns a structure 'data' where
  %   data.name           = name of the dataset
  %   data.filename       = unique name to use as a filename
  %   data.cache_filename = name to use for caches (can be shared for datasets that are subsets of domains)
  %   data.num_features   = number of features
  %   data.classes        = array of classes in the dataset
  %   data.domains        = cell array of names of the domains
  %   data.x              = cell array of training data for each domain
  %   data.y              = cell array of labels for each domain
  % 
  % Experimental setting information:
  %   data.domain_pairs             = pairs of [source,target] domain ids to include
  %   data.num_repetitions          = number of repetitions to perform
  %   data.train_idx                = indices to use for training in each repetition (optional)
  %   data.num_subsamples_per_class = number of instances per class to use from the source data (if set)
  %   data.preprocessing            = preprocessing strategy to use (as specified in argument, or default)
  % 
  % The cell arrays will have the same size.
  % Rows will be samples, columns the features
  
  % We assume that the data directory is located relative to the script
  datadir.base = 'data';
  datadir.ocvgg = '~/data/office-caltech';
  datadir.ovgg = '~/data/office31/vgg';
  datadir.decaf = '../data/';
  datadir.deep = '~/data/domain-adaptation';
  datadir.cdt = '../data/';
  
  if ~isstruct(data)
    data = struct('name',data);
    if nargin >= 2 && ~isempty(features)
      data.features = features;
    end
    if nargin >= 3 && ~isempty(preprocessing)
      data.preprocessing = preprocessing;
    end
  end
  
  % random seed for data shuffling
  rand('seed',1);
  
  if isequal(data.name, 'amazon')
    % Amazon dataset, full protocol
    data = load_amazon(datadir, data, 'full');
  elseif isequal(data.name, 'amazon-standard')
    data = load_amazon(datadir, data, 'standard');
  elseif isequal(data.name, 'amazon-repeated')
    data = load_amazon(datadir, data, 'full');
    data = shuffle_data(data);
    data.name = 'amazon-repeated';
    data.num_repetitions = 10;
  elseif isequal(data.name, 'amazon-subset')
    % Subset of the domains of the amazon data, as used in some papers
    data = load_amazon(datadir, data, 'full');
    data.name = 'amazon-subset';
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
  elseif isequal(data.name, 'amazon-standard-subset')
    data = load_amazon(datadir, data, 'standard');
    data.name = 'amazon-standard-subset';
    data.domain_pairs = [4 2; 2 1; 1 3; 3 4];
  
  elseif isequal(data.name, 'office-caltech')
    data = load_office_caltech(datadir, data, 'full');
  elseif isequal(data.name, 'office-caltech-repeated')
    data = load_office_caltech(datadir, data, 'full');
    data.name = 'office-caltech-repeated';
    data.num_repetitions = 10;
  elseif isequal(data.name, 'office-caltech-standard')
    data = load_office_caltech(datadir, data, 'standard');
    data = shuffle_data(data);
    
  elseif isequal(data.name, 'office')
    data = load_office(datadir, data, 'full');
  elseif isequal(data.name, 'office-31')
    data = load_office(datadir, data, 'full');
  elseif isequal(data.name, 'office-31-standard')
    data = load_office(datadir, data, 'standard');
  
  elseif isequal(data.name, 'cross-dataset-testbed')
    % Load the Cross-Dataset Testbed with decaf-fc7 features (subsampled)
    data = load_cross_dataset_testbed(datadir,data);
  
  elseif isfield(data, 'x')
    % assume already loaded
    warning('Calling load_data on already loaded dataset');
    
  else
    error('Unknown dataset: %s', data.name)
  end
  
  data.num_features = size(data.x{1},2);
  
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
end




function data = load_amazon(datadir, data, protocol)
  data.name = 'amazon';
  data.domains = {'books', 'dvd', 'elec', 'kitchen'};
  data.classes = [0,1];
  
  if ~isfield(data,'features')
    data.features = 'full';
  end
  
  if isequal(data.features, 'full')
    % Word frequency features
    % This is prepared from files from http://www.cs.jhu.edu/~mdredze/datasets/sentiment/
    data.cache_filename = data.name;
    data.display_name = '400 features';
    load([datadir.base, '/amazon.mat'], 'x','y','d');
    if ~isfield(data,'preprocessing')
      data.preprocessing = 'joint-std';
    end
    data.x = {};
    data.y = {};
    for i = 1:4
      data.y{i} = y(d==i,:);
      data.x{i} = x(d==i,:);
    end
  elseif isequal(data.features, '400')
    % Load the amazon dataset with 400 features,
    % from landmark paper; Gong, B.; Grauman, K.; and Sha, F.; 2013
    data.cache_filename = 'amazon-400';
    data.display_name = 'All features';
    if ~isfield(data,'preprocessing')
      data.preprocessing = 'zscore';
    end
    data.x = {};
    data.y = {};
    for i = 1:4
      load([datadir.base, '/amazon-400/',data.domains{i},'_400.mat'], 'fts', 'labels');
      data.x{i} = fts;
      data.y{i} = labels;
    end
  else
    error('Unknown features for amazon: %s', data.features);
  end
  
  if isequal(protocol,'full');
    data.num_repetitions = 1;
  else
    data.name = [data.name,'-standard'];
    data.cache_filename = [data.cache_filename,'-standard'];
    data.num_repetitions = 20;
    data.train_idx = {};
    data.test_idx = {};
    for i = 1:4
      load([datadir.base, '/amazon-400/',data.domains{i},'_400_idx.mat'], 'train_idx', 'test_idx');
      data.train_idx{i} = train_idx;
      data.test_idx{i} = test_idx;
    end
  end
end




function data = load_office_caltech(datadir, data, protocol)
  data.name = 'office-caltech';
  data.domains = {'amazon' 'Caltech10' 'dslr' 'webcam'};
  data.classes = 1:10;
  
  if ~isfield(data,'features')
    data.features = 'surf';
  end
  if isequal(data.features,'vgg')
    data.features = 'vgg-sumpool-fc6';
  end
  
  if isequal(data.features, 'surf')
    % Load the Office-Caltech datasets, with SURF features
    data.cache_filename = data.name;
    data.display_name = 'SURF features';
    if ~isfield(data,'preprocessing')
      data.preprocessing = 'norm-row,zscore';
    end
    data.x = {};
    data.y = {};
    for i=1:4
      load([datadir.base, '/office10/', data.domains{i}, '_SURF_L10.mat'], 'fts','labels');
      data.x{i} = fts;
      data.y{i} = labels;
    end
  elseif isequal(data.features, 'vgg-fc6')
    data = load_office_caltech_vgg(datadir,data,'fc6','');
  elseif isequal(data.features, 'vgg-fc7')
    data = load_office_caltech_vgg(datadir,data,'fc7','');
  elseif isequal(data.features, 'vgg-sumpool-fc6')
    data = load_office_caltech_vgg(datadir,data,'fc6','-sumpool');
    data.display_name = 'VGG-fc6 features';
  elseif isequal(data.features, 'vgg-sumpool-fc7')
    data = load_office_caltech_vgg(datadir,data,'fc7','-sumpool');
  elseif isequal(data.features, 'vgg-maxpool-fc6')
    data = load_office_caltech_vgg(datadir,data,'fc6','-maxpool');
  elseif isequal(data.features, 'vgg-maxpool-fc7')
    data = load_office_caltech_vgg(datadir,data,'fc7','-maxpool');
  elseif isequal(data.features, 'vgg16')
    data = load_office_caltech_deep(datadir,data,'vgg16');
    data.display_name = 'VGG16 features';
  elseif isequal(data.features, 'vgg19')
    data = load_office_caltech_deep(datadir,data,'vgg19');
    data.display_name = 'VGG19 features';
  elseif isequal(data.features, 'resnet50')
    data = load_office_caltech_deep(datadir,data,'resnet50');
    data.display_name = 'ResNet 50 features';
  elseif isequal(data.features, 'xception')
    data = load_office_caltech_deep(datadir,data,'xception');
    data.display_name = 'Xception V1 features';
  elseif isequal(data.features, 'inception_v3')
    data = load_office_caltech_deep(datadir,data,'inception_v3');
    data.display_name = 'Inception-v3 features';
  elseif isequal(data.features, 'inception_resnet_v2')
    data = load_office_caltech_deep(datadir,data,'inception_resnet_v2');
    data.display_name = 'Inception-ResNet-v2 features';
  elseif isequal(data.features, 'raw')
    % Don't actually load anything, this is included for results_from_papers that use raw features
    data.x = {[],[],[],[]};
    data.y = {[],[],[],[]};
    data.cache_filename = '';
    data.preprocessing = '';
  else
    error('Unknown features for office-caltech: %s', data.features);
  end
  
  if isequal(protocol,'full');
    data.num_repetitions = 1;
  else
    data.name = 'office-caltech-standard';
    data.cache_filename = [data.cache_filename,'-standard'];
    data.num_subsamples_per_class = [20,20,8,20];
    data.num_repetitions = 20;
  end
end

function data = load_office_caltech_vgg(datadir, data, layer, prefix)
  % Load the Office-Caltech datasets, with VGG features
  data.domains = {'amazon' 'Caltech' 'dslr' 'webcam'};
  data.cache_filename = sprintf('office-caltech-vgg%s-%s', prefix,layer);
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'truncate,joint-std';
  end
  data.x = {};
  data.y = {};
  for i=1:4
    load(sprintf('%s/office-caltech-vgg%s-%s-%s.mat', datadir.ocvgg, prefix, data.domains{i}, layer), 'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end

function data = load_office_caltech_deep(datadir, data, architecture)
  data.domains = {'amazon' 'Caltech' 'dslr' 'webcam'};
  data.cache_filename = sprintf('office-caltech-%s',architecture);
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'truncate,joint-std';
  end
  data.x = {};
  data.y = {};
  for i=1:4
    load(sprintf('%s/office-caltech-%s-%s.mat',datadir.deep, data.domains{i}, architecture),'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end




function data = load_office(datadir, data, protocol)
  data.name = 'office';
  data.domains = {'amazon' 'dslr' 'webcam'};
  data.classes = 1:31;
  
  if ~isfield(data,'features')
    data.features = 'decaf-fc7';
  end
  if isequal(data.features,'decaf')
    data.features = 'decaf-fc7';
  end
  if isequal(data.features,'vgg')
    data.features = 'vgg-sumpool-fc6';
  end
  
  if isequal(data.features, 'decaf-fc7')
    % Load the Office datasets, with decaf 7 features
    data = load_office_decaf(datadir,data,'decaf7');
    data.display_name = 'DECAF-fc7 features';
  elseif isequal(data.features, 'decaf-fc6')
    data = load_office_decaf(datadir,data,'decaf6');
  elseif isequal(data.features, 'vgg-fc6')
    data = load_office_vgg(datadir,data,'fc6','');
  elseif isequal(data.features, 'vgg-fc7')
    data = load_office_vgg(datadir,data,'fc7','');
  elseif isequal(data.features, 'vgg-fc8')
    data = load_office_vgg(datadir,data,'fc8','');
  elseif isequal(data.features, 'vgg-conv5')
    data = load_office_vgg(datadir,data,'conv5','');
  elseif isequal(data.features, 'vgg-sumpool-fc6')
    data = load_office_vgg(datadir,data,'fc6','-sumpool');
    data.display_name = 'VGG-fc6 features';
  elseif isequal(data.features, 'vgg-sumpool-fc7')
    data = load_office_vgg(datadir,data,'fc7','-sumpool');
  elseif isequal(data.features, 'vgg-maxpool-fc6')
    data = load_office_vgg(datadir,data,'fc6','-maxpool');
  elseif isequal(data.features, 'vgg-maxpool-fc7')
    data = load_office_vgg(datadir,data,'fc7','-maxpool');
  elseif isequal(data.features, 'vgg16')
    data = load_office_deep(datadir,data,'vgg16');
    data.display_name = 'VGG16 features';
  elseif isequal(data.features, 'vgg19')
    data = load_office_deep(datadir,data,'vgg19');
    data.display_name = 'VGG19 features';
  elseif isequal(data.features, 'resnet50')
    data = load_office_deep(datadir,data,'resnet50');
    data.display_name = 'ResNet 50 features';
  elseif isequal(data.features, 'xception')
    data = load_office_deep(datadir,data,'xception');
    data.display_name = 'Xception V1 features';
  elseif isequal(data.features, 'inception_v3')
    data = load_office_deep(datadir,data,'inception_v3');
    data.display_name = 'Inception V3 features';
  elseif isequal(data.features, 'inception_resnet_v2')
    data = load_office_deep(datadir,data,'inception_resnet_v2');
    data.display_name = 'Inception-ResNet V2 features';
  elseif isequal(data.features, 'raw')
    % Don't actually load anything, this is included for results_from_papers that use raw features
    data.x = {[],[],[]};
    data.y = {[],[],[]};
    data.cache_filename = '';
    data.preprocessing = '';
  else
    error('Unknown features for office: %s', data.features);
  end
  
  if isequal(protocol,'full');
    data.num_repetitions = 1;
  else
    % standard protocol: 30 labeled training examples
    data.name = [data.name,'-standard'];
    data.cache_filename = [data.cache_filename,'-standard'];
    data.num_subsamples_per_class = [30,30,30];
    data.num_repetitions = 5;
  end
end

function data = load_office_decaf(datadir,data,decaf)
  data.cache_filename = sprintf('office-%s',decaf);
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'zscore';
  end
  data.x = {};
  data.y = {};
  for i=1:3
    if true
      % use prepared mat files
      load(sprintf('%s/office_%s/%s.mat',datadir.decaf,decaf,data.domains{i}),'x','y');
      data.x{i} = x;
      data.y{i} = y;
    else
      path = sprintf('%s/office_%s/%s_%s/',datadir.decaf,decaf,data.domains{i},decaf);
      files = dir([path,'*.mat']);
      for j=1:numel(files)
        load([path,files(j).name]);
      end
      data.x{i} = 0
    end
  end
end

function data = load_office_vgg(datadir,data,layer,prefix)
  data.cache_filename = sprintf('office-vgg%s-%s',prefix,layer);
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'truncate,joint-std';
  end
  data.x = {};
  data.y = {};
  for i=1:3
    load(sprintf('%s/office-vgg%s-%s-%s.mat',datadir.ovgg,prefix,data.domains{i},layer),'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end

function data = load_office_deep(datadir, data, architecture)
  data.cache_filename = sprintf('office-31-%s',architecture);
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'truncate,joint-std';
  end
  data.x = {};
  data.y = {};
  for i=1:3
    load(sprintf('%s/office-31-%s-%s.mat',datadir.deep, data.domains{i}, architecture),'x','y');
    data.x{i} = double(x);
    data.y{i} = double(y');
  end
end





function data = load_cross_dataset_testbed(datadir,data)
  data.name = 'cross-dataset-testbed';
  data.domains = {'caltech256','imagenet','sun'};
  data.classes = 1:40;
  data.num_repetitions = 1;
  if ~isfield(data,'preprocessing')
    data.preprocessing = 'zscore';
  end
  
  if ~isfield(data,'features')
    data.features = 'decaf-fc7';
  end
  if isequal(data.features,'decaf-fc7')
    data.cache_filename = 'cross-dataset-testbed-decaf-fc7';
    data.x = {};
    data.y = {};
    for i=1:3
      load([datadir.cdt, '/Cross-Dataset-Testbed_SubSampled/dense_', data.domains{i}, '_decaf7_subsampled2.mat'], 'fts', 'labels');
      data.x{i} = fts;
      data.y{i} = labels;
    end
  else
    error('Unknown features for cross-dataset-testbed: %s', data.features);
  end
end



function data = shuffle_data(data)
  % Shuffle data for compatability with old cached results
  rand('seed',1);
  for i = 1:numel(data.domains)
    which = randperm(length(data.y{i}), length(data.y{i}));
    data.x{i} = data.x{i}(which,:);
    data.y{i} = data.y{i}(which,:);
  end
end

