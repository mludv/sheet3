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

%%
% constants
eta = 0.01; %learning rate
beta = 1/2;
layers = [2 3 2 1];
tmax = 10^6;
runs = 4;
Ht = zeros(tmax,runs);
Hv = zeros(tmax,runs);
Ct = zeros(tmax,runs);
Cv = zeros(tmax,runs);
parfor i = 1:runs
    disp(strcat('run ', num2str(i), ' started'));
    [ ~, Ht(:,i), Hv(:,i), Ct(:,i), Cv(:,i) ] = train_network( layers, train_data, valid_data, eta, beta, tmax );
    
    disp(strcat('run ', num2str(i), ' finished'));
end

%% Plot weights
t = linspace(-3, 1.5);
y = -w{1}(1)/w{1}(2)*t + w{1}(3)/w{1}(2);

hold on
plot(t, y)

