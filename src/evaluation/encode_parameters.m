function out = encode_parameters(x)
  % Encode method parameters as a string, to use as a filename for caching
  if iscell(x)
    out = '';
    for i=1:numel(x)
      if i>1, out = [out,'-']; end;
      out = [out,encode_parameters(x{i})];
    end
    % Output might become too large
    if length(out)>150
      out = [out(1:100), md5sum(out,true)];
    end
  elseif isstruct(x)
    out = '';
    fields = fieldnames(x);
    for i=1:numel(fields)
      if i>1, out = [out,'-']; end;
      out = [out,fields{i},'=',encode_parameters(getfield(x,fields{i}))];
    end
  elseif ischar(x)
    out = x;
  elseif isnumeric(x) && ~isscalar(x)
    out = '';
    for i=1:numel(x)
      if i>1, out = [out,',']; end;
      out = [out,encode_parameters(x(i))];
    end
  elseif isscalar(x) && isnumeric(x)
    out = sprintf('%g',x);
  elseif isscalar(x) && islogical(x)
    out = sprintf('%d',x);
  elseif isa(x,'function_handle')
		info = functions(x);
		out = info.function;
  else
    error('Don''t know how to encode parameter %s', disp(x));
  end
end
