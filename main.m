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

% initialize weights and thresholds
w = 0.2*rand(1,2);
theta = 2*rand - 1;

% train network
tmax = 10^6;
for i = 1:tmax
    % choose a pattern
    zeta = datasample(train_data,1);
    
    b = w * zeta([1 2])'-theta;
    output = g(b);
    
    error = dg(b)*(zeta(3)-output);
    
    w = w + eta*error*zeta([1 2]);
    theta = theta-eta*error;
    
end
    
%% Plot weights
t = linspace(-3, 1.5);
y = -w(1)/w(2)*t - theta/w(2);

hold on
plot(t, y)

