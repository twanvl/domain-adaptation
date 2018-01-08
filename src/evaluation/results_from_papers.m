function results = results_from_papers(data)
  % Results copied from papers
  
  % [1] Return of Frustratingly Easy Domain Adaptation;
  %     Baochen Sun, Jiashi Feng, Kate Saenko;
  %     AAAI 2016
  
  results = struct();
  results.methods = {};
  results.accs = [];
  
  if isequal(data.name, 'office-caltech-standard') && isequal(data.features,'surf') && isequal(data.preprocessing,'norm-row,zscore')
    % Table 1 of [1]: Office-caltech, standard protocol
    % NA = no adaptation
    results.methods = {'NA','SVMA','DAM','GFK','TCA','CORAL'};
    results.accs    = ...
      [35.8 34.8 34.9 38.3 40.0 39.9 40.3 ... % A→C
      ;33.1 34.1 34.3 37.9 39.1 38.8 38.3 ... % A→D
      ;24.9 32.5 32.5 39.8 40.1 39.6 38.7 ... % A→W
      ;43.7 39.1 39.2 44.8 46.7 46.1 47.2 ... % C→A
      ;39.4 34.5 34.7 36.1 41.4 39.4 40.7 ... % C→D
      ;30.0 32.9 33.1 34.9 36.2 38.9 39.2 ... % C→W
      ;26.4 33.4 33.5 37.9 39.6 42.0 38.1 ... % D→A
      ;27.1 31.4 31.5 31.4 34.0 35.0 34.2 ... % D→C
      ;56.4 74.4 74.7 79.1 80.4 82.3 85.9 ... % D→W
      ;32.3 36.6 34.7 37.1 40.2 39.3 37.8 ... % W→A
      ;25.7 33.5 31.2 29.1 33.7 31.8 34.6 ... % W→C
      ;78.9 75.0 68.3 74.6 77.5 77.9 84.9 ... % W→D
      ;37.8 41.0 40.2 43.4 45.7 45.9 46.7 ... % avg
      ]' / 100;
    %results.methods{end+1} = 
    %results.accs(:,end+1)  = 
    
  elseif isequal(data.name, 'office')
    % Table 2 of [1]: Office31
    office_31_methods = {'NA-fc6','NA-fc7','NA-FT6','NA-FT7','SA-fc6','SA-fc7','SA-FT6','SA-FT7','GFK-fc6','GFK-fc7','GFK-FT6','GFK-FT7','TCA-fc6','TCA-fc7','TCA-FT6','TCA-FT7','DLID','DANN','DA-NBNN','DECAF-fc6','DECAF-fc7','DDC','DAN','ReverseGrad','CORAL-fc6','CORAL-fc7','CORAL-FT6','CORAL-FT7'};
    office_31_results = ...
      [53.2 55.7 54.5 58.5 41.3 46.2 40.5 50.5 44.8 52 48.8 56.4 40.6 45.4 40.8 47.3 nan 34.0 nan nan nan nan nan nan 53.7 57.1 61.2 62.2 ... % A→D
      ;48.6 50.6 48.0 53.0 35 42.5 41.1 47.2 37.8 48.2 45.6 52.3 36.8 40.5 37.2 45.2 26.1 34.1 23.3 52.2 53.9 59.4 66.0 67.3 48.4 53.1 59.8 61.9 ... % A→W
      ;40.5 46.5 38.9 43.8 32.3 39.3 33.8 39.6 34.8 41.8 40.5 43.2 32.9 36.5 30.6 36.4 nan 20.1 nan nan nan nan nan nan 44.4 51.1 47.4 48.4 ... % D→A
      ;92.9 93.1 91.2 94.8 74.5 78.9 85.4 89 81 86.5 90.4 92.2 82.3 78.2 79.5 80.9 68.9 62.0 67.2 91.5 89.2 92.5 93.5 94.0 96.5 94.6 97.1 96.2 % D→W
      ;39.0 43.0 40.7 43.7 30.1 36.3 33.4 37.3 31.4 38.6 36.7 41.5 28.9 34.1 36.7 39.2 nan 21.2 nan nan nan nan nan nan 41.9 47.3 45.8 48.2 ... %W→A
      ; 98.8 97.4 98.9 99.1 81.5 80.6 88.2 93 86.9 87.5 96.3 96.6 84.1 84 91.8 92 84.9 64.4 67.4 nan nan 91.7 95.3 93.7 99.2 98.2 99.5 99.5 ... % W→D
      ; 62.2 64.4 62.0 65.5 49.1 54.0 53.7 59.4 49.1 59.1 59.7 63.7 50.9 53.1 52.8 56.8 nan 39.3 nan nan nan nan nan nan 64.0 66.9 68.5 69.4 % AVG
      ];
    if isequal(data.features,'decaf-fc6') && isequal(data.preprocessing,'zscore')
      which = cellfun(@(x)isequal(x(end-2:end),'fc6'),office_31_methods) & ~any(isnan(office_31_results));
    elseif isequal(data.features,'decaf-fc7') && isequal(data.preprocessing,'zscore')
      which = cellfun(@(x)isequal(x(end-2:end),'fc7'),office_31_methods) & ~any(isnan(office_31_results));
    else
      which = [];
    end
    results.methods = cellfun(@(x)x(1:end-4),office_31_methods(which),'UniformOutput',false);
    results.accs = office_31_results(:,which)' / 100;
    
    % Other results on office dataset
    % From TDS paper
    if isequal(data.features,'raw')
      results.methods{end+1} = 'DLID'; %  (Chopra et al., 2013)
      results.accs(end+1,:)  = [nan, 51.9, nan, 78.2, nan, 89.9, nan] / 100;
      results.methods{end+1} = 'DDC'; %  (Tzeng et al., 2014)
      results.accs(end+1,:)  = [64.4, 61.8, 52.1, 95.0, 52.2, 98.5, 70.7] / 100;
      results.methods{end+1} = 'DAN'; %  (Long and Wang, 2015)
      results.accs(end+1,:)  = [67.0, 68.5, 54.0, 96.0, 53.1, 99.0, 72.9] / 100;
      results.methods{end+1} = 'DANN';
      results.accs(end+1,:)  = [nan, 73.0, nan, 96.4, nan, 99.2, nan] / 100;
      results.methods{end+1} = 'BP';
      results.accs(end+1,:)  = [72.8, 73.0, 54.4, 96.4, 53.6, 99.2, 74.9] / 100;
      results.methods{end+1} = 'TDS';
      results.accs(end+1,:)  = [84.1, 81.1, 58.3, 96.4, 63.8, 99.2, 80.5] / 100;
    end
  elseif isequal(data.name, 'office-caltech') && isequal(data.features,'surf') && isequal(data.preprocessing,'norm-row,zscore')
    % Table 3 of [1]: Office-caltech, full
    results.methods = {'NA','SA','GFK','TCA','CORAL'};
    results.accs = ...
      [41.7 37.4 41.9 35.2 45.1 ... % A→C
      ;44.6 36.3 41.4 39.5 39.5 ... % A→D
      ;31.9 39.0 41.4 29.5 44.4 ... % A→W
      ;53.1 44.9 56.0 46.8 52.1 ... % C→A
      ;47.8 39.5 42.7 52.2 45.9 ... % C→D
      ;41.7 41.0 45.1 38.6 46.4 ... % C→W
      ;26.2 32.9 38.7 36.2 37.7 ... % D→A
      ;26.4 34.3 36.5 30.1 33.8 ... % D→C
      ;52.5 65.1 74.6 71.2 84.7 ... % D→W
      ;27.6 34.4 31.9 32.2 36.0 ... % W→A
      ;21.2 31.0 27.5 27.9 33.7 ... % W→C
      ;78.3 62.4 79.6 74.5 86.6 ... % W→D
      ;41.1 41.5 46.4 42.8 48.8 ... % AVG
      ]' / 100;
    
  elseif isequal(data.name, 'cross-dataset-testbed') && isequal(data.features,'decaf-fc7') && isequal(data.preprocessing,'zscore')
    % Table 4 of [1]: Cross dataset testbed
    results.methods = {'NA','SA','GFK','TCA','CORAL'};
    results.accs = ...
      [66.1 43.7 52   48.6 66.2 ... % C→I
      ;21.9 13.9 18.6 15.6 22.9 ... % C→S
      ;73.8 52.0 58.5 54.0 74.7 ... % I→C
      ;22.4 15.1 20.1 14.8 25.4 ... % I→S
      ;24.6 15.8 21.1 14.6 26.9 ... % S→C
      ;22.4 14.3 17.4 12.0 25.2 ... % S→I
      ;38.5 25.8 31.3 26.6 40.2 ... % AVG
      ]' / 100;
    
  elseif isequal(data.name, 'amazon-standard-subset') && isequal(data.features,'400') && isequal(data.preprocessing,'zscore')
    % Table 5 of [1]: Amazon
    results.methods = {'NA','TCA','SA','GFS','GFK','SCL','KMM','CORAL'};
    results.accs = ...
      [72.2 60.4 78.4 67.9 69.0 72.8 72.2 73.9 ... % K→D
      ;76.9 61.4 74.7 68.6 71.3 76.2 78.6 78.3 ... % D→B
      ;74.7 61.3 75.6 66.9 68.4 75.0 76.9 76.3 ... % B→E
      ;82.8 68.7 79.3 75.1 78.2 82.9 83.5 83.6 ... % E→K
      ;76.7 63.0 77.0 69.6 71.7 76.7 77.8 78.0 ... % AVG
      ]' / 100;
    
  elseif isequal(data.name, 'imdb')
    % Results from http://jmlr.org/papers/volume17/15-206/15-206.pdf
    results.methods = {'S-SVM','S-LR','KMM','SCL','SA','GFK','TCA','FLDA-Q','FLDA-L','T-LR'};
    results.accs = ...
      [.145,.136,.133,.133,.184,.276,.230,.135,.136,.196 ... % A→F
      ;.158,.155,.155,.165,.163,.249,.266,.158,.154,.163 ... % A→W
      ;.256,.206,.208,.206,.182,.289,.355,.205,.202,.163 ... % F→W
      ;.201,.195,.193,.198,.193,.296,.363,.194,.194,.169 ... % F→A
      ;.168,.160,.159,.159,.167,.238,.222,.155,.157,.169 ... % W→A
      ;.340,.167,.163,.163,.232,.292,.203,.172,.159,.196 ... % W→F
      ]';
    
  else
    results.methods = {};
    results.accs = [];
  end
end

