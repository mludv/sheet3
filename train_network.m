function [ w, Ht, Hv, Ct, Cv ] = train_network( layers, train_data, valid_data, eta, beta, tmax )
%TRAIN_NETWORK Summary of this function goes here
%   Detailed explanation goes here

% constants
g = @(x) tanh(beta*x);     % activation function
dg = @(x) beta*(1-g(x).^2); % derivative of g
numPatterns = size(train_data,1);
numValidPat = size(valid_data,1);


% add a -1 to the data sets so we can subtract the threshold efficently
train = [train_data(:,[1 2]) -1.*ones(numPatterns,1)];
valid = [valid_data(:,[1 2]) -1.*ones(numValidPat,1)];

% initialize weights and thresholds
w = cell(length(layers)-1,1);
for i = 1:(length(layers)-1)
    rows = layers(i+1);
    cols = layers(i);
    w{i} = 0.4*rand(rows,cols) - 0.2;
    w{i}(:,cols+1)   = 2*rand(rows,1) - 1;
end


% train network
Ht = zeros(1,tmax);
Hv = zeros(1, tmax);
Ct = zeros(1,tmax);
Cv = zeros(1, tmax);

for i = 1:tmax
    % choose a pattern
    [pattern, index] = datasample(train,1);
    real_output = train_data(index,3);
    
    
    % propagate forward
    M = length(layers); % number of layers
    V = cell(M,1);
    V{1} = pattern';
    for m = 1:M-1
        V{m+1} = g(w{m}*V{m});
        V{m+1}(end+1) = -1;
    end
    
    % calculate error for output layer
    output_train = V{M}(1:end-1); % remove the last "-1"
    error = cell(M,1);
    error{M} = dg(w{M-1}*V{M-1})*(real_output-output_train);
    
    % calculate errors for hidden layers
    for m = M:-1:3
        error{m-1} = dg(w{m-2}*V{m-2}).*(w{m-1}(:,1:end-1)'*error{m});
    end
    
    % update weights
    for m = 1:M-1
        w{m} = w{m}+eta.*error{m+1}*V{m}';
    end
    
    % calculate energies
    
    output_train = cell(M,1);
    output_train{1} = train;
    output_valid = cell(M,1);
    output_valid{1} = valid;
    for m=1:M-1
        output_train{m+1} = g(w{m}*output_train{m}')';
        output_valid{m+1} = g(w{m}*output_valid{m}')';
        
        % put -1 in a new column if it is not the final output
        if (m~=M-1)
            output_train{m+1}(:, end+1) = -1.*ones(numPatterns,1);
            output_valid{m+1}(:, end+1) = -1.*ones(numValidPat,1);
        end
    end
    
    %save energies
    Ht(i) = 0.5 * sum((train_data(:, 3) - output_train{M}).^2);
    Ct(i) = 0.5/numPatterns * sum(abs(train_data(:, 3) - sign(output_train{M})));
    Hv(i) = 0.5 * sum((valid_data(:, 3) - output_valid{M}).^2);
    Cv(i) = 0.5/numValidPat * sum(abs(valid_data(:, 3) - sign(output_valid{M})));
    
    if rem(i, 10000) == 0
        disp(strcat(num2str(i/10000), '/', num2str(tmax/10000), ' finished'));
    end
end


end

