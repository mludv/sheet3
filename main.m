% Sheet 3

%% Import data
train_data = dlmread('train_data1.txt');
valid_data = dlmread('valid_data1.txt');

% normalize

my = mean(train_data);
sigma = std(train_data);

train_data(:,1) = (train_data(:,1)-my(1))./sigma(1);
train_data(:,2) = (train_data(:,2)-my(2))./sigma(2);
valid_data(:,1) = (valid_data(:,1)-my(1))./sigma(1);
valid_data(:,2) = (valid_data(:,2)-my(2))./sigma(2);

%% Plot validation data
A = valid_data(:,3) == 1;
B = valid_data(:,3) == -1;
clf
hold on
scatter(valid_data(A, 1), valid_data(A,2));
scatter(valid_data(B, 1), valid_data(B,2));
xlim([-3, 1.5])

%% Plot training data
A = train_data(:,3) == 1;
B = train_data(:,3) == -1;
clf
hold on
scatter(train_data(A, 1), train_data(A,2));
scatter(train_data(B, 1), train_data(B,2));
hold off


%% 2a)

% constants
eta = 0.01; %learning rate
beta = 1/2;
g = @(x) tanh(beta*x);     % activation function
dg = @(x) beta*(1-g(x).^2); % derivative of g
numPatterns = size(train_data,1);
numValidPat = size(valid_data,1);


% add a -1 to the data sets so we can subtract the threshold efficently
train = [train_data(:,[1 2]) -1.*ones(numPatterns,1)];
valid = [valid_data(:,[1 2]) -1.*ones(numValidPat,1)];

% the layers and how many neurons
layers = [2 1];

% initialize weights and thresholds
w = cell(length(layers)-1,1);
for i = 1:(length(layers)-1)
    rows = layers(i+1);
    cols = layers(i);
    w{i} = 0.4*rand(rows,cols) - 0.2;
    w{i}(:,cols+1)   = 2*rand(rows,1) - 1;
end


% train network
tmax = 10^4;
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
    output = V{M}(1:end-1); % remove the last "-1"
    error = cell(M,1);
    error{M} = dg(w{M-1}*V{M-1})*(real_output-output);
    
    % calculate errors for hidden layers
    for m = M:-1:3
        error{m-1} = dg(w{m-2}*V{m-2}).*(w{m-1}(:,1:end-1)'*error{m});
    end
    
    % update weights
    for m = 1:M-1
        w{m} = w{m}+eta.*error{m+1}*V{m}';
    end
    
    %save energies
    %Ht(i) = 0.5/numPatterns * sum((train_data(:, 3)' - g(w{M-1}*train')).^2);
    %Ct(i) = 0.5/numPatterns * sum(abs(train_data(:, 3)' - g(w{M-1}*train')));
    %Hv(i) = 0.5/numValidPat * sum((valid_data(:, 3)' - g(w{M-1}*valid')).^2);
    %Cv(i) = 0.5/numValidPat * sum(abs(valid_data(:, 3)' - g(w{M-1}*valid')));
    
    if rem(i, 10000) == 0
        disp(i/10000)
    end
end
    
%% Plot weights
t = linspace(-3, 1.5);
y = -w{M-1}(1)/w{M-1}(2)*t + w{M-1}(3)/w{M-1}(2);

hold on
plot(t, y)

