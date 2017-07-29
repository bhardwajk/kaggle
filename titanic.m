%{ 
Steps:
1. Process data - copy values of continuous features; for discrete features (like gender, class) create appropriate number of additional features
2. Normalize data
2. Run logistic regression as the final outcome is boolean (survived or not). Fminunc!
3. Split training data into train + cross validation to improve algo
4. Run diagnostics to see bias-variance tradeoffs
%}

pkg load io;

function output = cell2mat_special (input)

    if length(cell2mat(input)) == 0
        output = 0;
    else 
        output = cell2mat (input); 
    endif

endfunction

function [code, number] = splitCodeNumberFromTicket (str)
    code = '';
    number = 0;

    if ischar (str) 
        spacePos = strchr(str, " ");
        if length(spacePos) == 0
            % No spaces, and has some text => full ticket str is code
            code = str;
        else
            % Find position of last space, split str at that position. First half is code, next half is number
            code = str(1:spacePos(end));
            number = str2num(str(spacePos(end)+1:length(str)));
        endif
    else
        number = str;
    endif
endfunction

function cabinNumbers = getCabinNumbers (str)
    % Example - if str = "C34 A55 D67 C20", this function will return an array
    % [1 0 2 1 0 0 ... 0]
    % i.e., number of rooms in each letter. Though we are loosing info about room number. 

    cabinNumbers = zeros(1, 26);
    if str == 0
        return;
    endif

    for split = strsplit(str)
        s = cell2mat(split);
        if length(s) > 1
            cn = s(1) - "A" + 1; 
            cabinNumbers(cn) += 1; %str2num(s(2:end));
        endif
    endfor
endfunction

function [X,y] = parseinputs (filename, isTestData)

    inputs = csv2cell(filename); 
    inputs = inputs (2:end, :); %Skip first line (header)

    m = size(inputs)(1); % number of entries/rows
    n = size(inputs)(2); % number of features/columns
    
    y = zeros (m, 1);

    % Fill up input & output matrix column-wise
    % Couldn't find an elegant way to convert cellArray into Matrix, hencing using iteration
    
    % Order of inputs for Training Data
    % PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    % Order of inputs for Test Data
    % PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

    if isTestData == 0
        % Column Survived - copy to y (output). Dont fill this in X
        for i = 1:m
            y(i,1) = cell2mat_special(inputs(i,2));
        end

        % Remove column #2 so that now test & training data's format is same
        inputs = inputs (:, [1,3,4,5,6,7,8,9,10,11,12]);
    end
    %X = inputs;
    %return;

    %ticketCodes = getTicketCodes (inputs(:,8));
    
    for i = 1:m
    
        % Column PassengerId - copy as it is
        X(i,1) = cell2mat_special(inputs(i,1));
        
        % Column Pclass 
        cls = cell2mat_special(inputs(i,2));
        X(i,[2 3 4]) = 0;
        X(i, 1+cls) = 1;
        
        % Column Name - not sure how name can affect surviving chances. Hence skipping it. Maybe we can come to it later. 
        % Ignore inputs(:,3) for now. 
        X(i,5) = 0;
        
        % Column sex - convert to boolean (male = 1, female = 0)
        X(i,6) = strcmp(inputs(i,4), "male");
        X(i,7) = strcmp(inputs(i,4), "female");
        
        % Column Age - copy as it is
        X(i,8) = cell2mat_special(inputs(i,5));

        % Column Sibsp - copy as it is
        X(i,9) = cell2mat_special(inputs(i,6));

        % Column Parch - copy as it is
        X(i,10) = cell2mat_special(inputs(i,7));

        % Column Tickets - each ticket consists of 2 words - first is text, second is numeric. 
        % We will split ticket id into 2 columns and encode each text to a certain number. 
        % Example, Ticket "A/5 21171" will be split into two features -> X1 = 10 (some encoding for A/5 text) & X2 = 21171
        str = cell2mat_special(inputs(i,8));
        [tktCode, tktNumber] = splitCodeNumberFromTicket (str);
        X(i, 11) = 0; %ticketCodes.tktCode
        X(i, 12) = tktNumber;


        % Column Fare - copy as it is
        X(i,13) = cell2mat_special(inputs(i,9));

        % Column Cabin - since columns start from A to G letters, keeping 26 features for it. Each value will be room number
        X(i,[14:39]) = getCabinNumbers(cell2mat_special(inputs(i,10)));

        % Column Embarked - 1 for Cherbourg, 2 for Queenstown, 3 for Southampton
        X(i, [40 41 42]) = 0;
        e = cell2mat_special(inputs(i,11));
        if strncmp (e, "C", 1)
            X(i,40) = 1;
        elseif strncmp (e, "Q", 1)
            X(i,41) = 1;
        elseif strncmp (e, "S", 1)
            X(i,42) = 1;
        endif
    endfor


    % Normalize continuous values
    % PassengerID, Age, Sisb, Parch, tktNumber, CabinNumber
    for i = [1 8 9 10 12 13 14:39]
        if max(X(:,i))==min(X(:,i))
            X(:,i) = (X(:,i) - mean(X(:,i)));
        else
            X(:,i) = (X(:,i) - mean(X(:,i)))/(max(X(:,i))-min(X(:,i)));
        endif
    endfor
endfunction

function g = sigmoid(z)
    g = 1 ./ (1 + exp (-1*z));
endfunction

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y); % number of training examples
    
    J = 0;
    grad = zeros(size(theta));
    
    n = size(theta);
    temp = eye(n, n);
    temp(1,1) = 0; % to eliminate theta(0) in calculation of grad
    
    J = (1/m) * sum ( -y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta))) + ...
        (lambda*0.5/m) * sum (theta(2:n) .^ 2);
    grad = (1/m) * X' * (sigmoid(X*theta) - y) + (lambda/m)*temp*theta;

endfunction


[X,y] = parseinputs('train.csv', 0);

% Split training & validation set
m = length(X);
trainLength = round(m*0.7); %Split between training & cross validation data

Xtrain = X(1:trainLength, :);
ytrain = y(1:trainLength, :);
Xcv = X(trainLength+1:end, :);
ycv = y(trainLength+1:end, :);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 4000);

initial_theta = zeros(size(Xtrain, 2), 1);
lambda = 0;

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, Xtrain, ytrain, lambda)), initial_theta, options);

fprintf ('Error in training set: %f\n', J);
fprintf ('Accuracy in training set: %f\n', sum(sigmoid(Xtrain*theta)>0.5 == ytrain)/length(ytrain));
fprintf ('Error in cross validation set: %f\n', costFunctionReg(theta, Xcv, ycv, lambda)(1));
fprintf ('Accuracy in training set: %f\n', sum(sigmoid(Xcv*theta)>0.5 == ycv)/length(ycv));


% Now print output of test.csv in expected format
[Xtest ytest] = parseinputs('test.csv', 1);
ytest = sigmoid(Xtest*theta) > 0.5;
passengerIdTestSet = [892:1309]'; %Hard coding test.csv passenger id

%Append timestamp to create new output fileeach time. 
outputFileName = strcat('output', num2str(round(time)), '.csv');
fid = fopen(outputFileName, 'w');
fprintf (fid, 'PassengerId,Survived\n');
dlmwrite (fid, [passengerIdTestSet ytest]);
fclose(fid);

