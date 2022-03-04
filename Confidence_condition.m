function [Cc] = Confidence_condition(p_bar,f_current)

p_bar;

n_ker = length(p_bar);
value = zeros(n_ker,1);


for j = 1:n_ker 
    
    ((f_current - (ones(size(f_current))*f_current(j))).^2);
    value(j) = p_bar'*((f_current - (ones(size(f_current))*f_current(j))).^2);
        
end

Cc = max(value);

end