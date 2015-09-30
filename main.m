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

%% Plot training data
A = train_data(:,3) == 1;
B = train_data(:,3) == -1;
clf
hold on
scatter(train_data(A, 1), train_data(A,2));
scatter(train_data(B, 1), train_data(B,2));
hold off

%% Plot validation data
A = valid_data(:,3) == 1;
B = valid_data(:,3) == -1;
clf
hold on
scatter(valid_data(A, 1), valid_data(A,2));
scatter(valid_data(B, 1), valid_data(B,2));
xlim([-3, 1.5])

%% 2a)

% constants
eta = 0.01; %learning rate
beta = 1/2;
g = @(x) tanh(beta*x);     % activation function
dg = @(x) beta*(1-g(x)^2); % derivative of g
numPatterns = size(train_data,1);
numValidPat = size(valid_data,1);

% initialize weights and thresholds
w = 0.4*rand(1,2) - 0.2;
w(3)   = 2*rand - 1;

% add a -1 to the data sets so we can subtract the threshold efficently
train = [train_data(:,[1 2]) -1.*ones(numPatterns,1)];
valid = [valid_data(:,[1 2]) -1.*ones(numValidPat,1)];

% train network
tmax = 10^6;
Ht = zeros(1,tmax);
Hv = zeros(1, tmax);
Ct = zeros(1,tmax);
Cv = zeros(1, tmax);
%
for i = 1:tmax
    % choose a pattern
    [pattern, index] = datasample(train,1);
    real_output = train_data(index,3);
    
    % calculate output
    b = w * pattern';
    output = g(b);
    
    % calculate error
    error = dg(b)*(real_output-output);
    
    % update weights
    w = w + eta*error*pattern;
    
    % save energies
    Ht(i) = 0.5/numPatterns * sum((train_data(:, 3)' - g(w*train')).^2);
    Ct(i) = 0.5/numPatterns * sum(abs(train_data(:, 3)' - g(w*train')));
    Hv(i) = 0.5/numValidPat * sum((valid_data(:, 3)' - g(w*valid')).^2);
    Cv(i) = 0.5/numValidPat * sum(abs(valid_data(:, 3)' - g(w*valid')));
    
    if rem(i, 10000) == 0
        disp(i/10000)
    end
end
    
%% Plot weights
t = linspace(-3, 1.5);
y = -w(1)/w(2)*t - w(3)/w(2);

hold on
plot(t, y)

