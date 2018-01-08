function methods = dummy_methods(varargin)
  % Methods for which to use results_from_papers
  if nargin == 0
    names = {'GFK','SA','CORAL'};
  else
    names = varargin;
  end
  
  methods = {};
  for i=1:numel(names)
    methods{end+1} = struct('dummy',true, 'name',names{i});
  end
end
