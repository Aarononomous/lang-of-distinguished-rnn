classdef RNN < handle
    properties (Constant)
        report_header =  ['        | TRAIN                        TEST                         VALIDATION\n' ...
            '   Ep   | Cost  |   Corr    | Acc   || Cost  |   Corr    | Acc   || Cost  |   Corr    | Acc\n'];
        
        report_format = ' %4d:\t| %1.3f | %4d/%4d | %1.3f || %1.3f | %4d/%4d | %1.3f || %1.3f | %4d/%4d | %1.3f\n';
    end
    
    properties
        corpus;
        W; % weights from hidden layers in gen. t-1 to gen t
        U; % weights from inputs to hidden layers
        b; % biases for hidden layer
        V; % weights from hidden layer to output
        c; % biases for output layer
        isLearning = false;
        activation_function = @RNN.ReLU;
        activation_function_prime = @RNN.ReLU_prime;
        cost_function = @RNN.cross_entropy;
        cost_function_prime = @RNN.cross_entropy_prime;
        regularization_function = @RNN.L2;
        regularization_function_prime = @RNN.L2_prime;
        regularization_factor = 0.05;
        training_stats = [];
        % momentum = 0.0;
    end
    
    properties (Dependent)
        L;
        input_neurons;
        output_neurons;
    end
    
    methods (Static)
        %% Helper functions
        function out = one_hot(vector, elems)
            % creates a one-hot array of length size with the element at
            % pos = 1
            len = size(elems, 2);
            out = zeros(vector, len);
            for p = 1 : len
                out(elems(p), p) = 1;
            end
        end
        
        function out = max_n(a, n)
            % returns the n largest elements of a cell array in sorted
            % order
            % if n is multi-dimensional, this is ordered by the first dim
            n = min(n, size(a, 1));
            out = sortrows(a);
            out = out(end: -1: end - n, :);
        end
        
        function [a, b, c] = split_input(arr, part_a, part_b, part_c)
            % splits input into three parts a, b, and c by columns
            % the parts should sum to 1
            a_end = floor(size(arr, 2) * part_a);
            b_end = floor(size(arr, 2) * (part_a + part_b));
            c_end = floor(size(arr, 2) * (part_a + part_b + part_c));
            a = arr(:, 1 : a_end);
            b = arr(:, (a_end + 1) : b_end);
            c = arr(:, (b_end + 1) : c_end);
        end
        
        function out = shuffle(cell_arr)
            shuffled_order = randperm(size(cell_arr, 2));
            out = cell_arr(:, shuffled_order);
        end
        
        %% Activation functions
        
        % sigmoid
        function out = sigmoid(t, isLast)
            out = 1 ./ (1 + exp(-t));
        end
        
        function out = sigmoid_prime(t)
            s = sigmoid(t);
            out = s .* (1 - s);
        end
        
        % tanh
        function out = tanh(t, isLast)
            out = tanh(t);
        end
        
        function out = tanh_prime(t)
            out = log(cosh(t));
        end
        
        % softmax
        function out = softmax(t, isLast)
            if isLast
                out = exp(t) ./ sum(exp(t));
            else
                out = RNN.sigmoid(t, false);
            end
        end
        
        function out = softmax_prime(t)
            out = RNN.sigmoid_prime(t);
        end
        
        % ReLU
        function out = ReLU(t, isLast)
            out = max(t, 0);
        end
        
        function out = ReLU_prime(t)
            out = t > 0;
        end
        
        %% Cost functions
        
        % MSE (Quadratic)
        function out = MSE(target, actual)
            % Calculate the mean squared error:
            % C(w, b) = 1/2n * Sum( ||(y - a)||^2 )
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
            lengths = sqrt(sum((target - actual) .^ 2));
            out = sum(lengths .^ 2) / (2 * size(target, 2));
        end
        
        function out = MSE_prime(target, actual, z, afp)
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
            out = (actual - target) .* afp(z);
        end
        
        % Cross-entropy
        function out = cross_entropy(target, actual)
%                 target
%               actual
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
%             costRets = - ((sum(sum(((targets.*log(activations)) + ...
%                         (1-targets).*log(1-activations)),2))) ...
%               / setSize) + (lambda*sumSqWts/(2*numInstances));

            out = -sum(sum(target .* log(actual) + (1 - target) .* log(1 - actual))) / size(target, 2);
        end
        
        function out = cross_entropy_prime(target, actual, z, afp)
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
            out = actual - target;
        end
        
        % Log-likelihood
        function out = log_likelihood(target, actual)
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
            sum(log(sum(actual .* target)));
            out = 0;
        end
        
        function out = log_likelihood_prime(target, actual, z, afp)
            if (size(target) ~= size(actual))
                error('The cost function needs equal-sized arrays.\n');
            end
            
            out = actual - target;
        end
        
        %% Regularization functions
        %% The regularization functions are applied to all the weights in the net,
        %% the gradients are applied to individual weights (one layer at a time)
        
        % no regularization
        function out = none(~, ~)
            out = 0;
        end
        
        % L1 regularization = lambda/2n * Sum (w)
        % divide by n later
        function out = L1(lambda, weights)
            sum_weights = sum(cell2mat(cellfun(@(x) sum(sum(x)), weights, 'UniformOutput', false)));
            out = (lambda / 2) * sum_weights;
        end
        
        % L1' = lambda/2n
        % divide by n later
        function out = L1_prime(lambda, weights)
            out = lambda / 2;
        end
        
        %% L2 regularization = lambda/2n * Sum (w^2)
        % divide by n later
        function out = L2(lambda, weights)
            % sum_squares = sum(cell2mat(cellfun(@(x) sum(sum(x .^ 2)), weights, 'UniformOutput', false)))
            sum_squares = 0;
            for l = 1 : size(weights, 1)
                sum_squares = sum_squares + sum(sum(weights{l} .^ 2));
            end
            out = (lambda / 2) * sum_squares;
        end
        
        %% L2' = lambda/n * w
        % divide by n later
        function out = L2_prime(lambda, weights)
            out = lambda * weights;
        end
    end
    
    methods
        % Constructor
        function obj = RNN(W, U, b, V)
            %%% TODO: what parameters? A: Cells of obj.weights and obj.biases
            if nargin == 4
                obj.W = W;
                obj.U = U;
                obj.b = b;
                obj.V = V;
            end
        end
                
        function initialize_weights(obj, hidden_layer_size)
            if (isempty(obj.corpus))
                error('Can''t initialize weights without corpus specified.');
            end
            
            % the size of the input layer is also the variance
            variance = size(obj.corpus.allChars, 2);
            
            % initialize W, weights from hidden layers in gen. t-1 to gen t
            obj.W = randn(hidden_layer_size, hidden_layer_size) / sqrt(variance);
            
            % initialize U, weights from inputs to hidden layers
            obj.U = randn(hidden_layer_size, variance) / sqrt(variance);
            
            % initialize b, biases for hidden layer
            obj.b = randn(hidden_layer_size, 1) / sqrt(variance);
            
            % initialize V, weights from hidden layer to output
            obj.V = randn(size(obj.corpus.languages, 2), hidden_layer_size) / sqrt(variance);
            
            % initialize c, biases for output layer
            obj.c = randn(size(obj.corpus.languages, 2), 1) / sqrt(variance);
            
            % reset training stats
            obj.training_stats = [];
        end
        
        function out = feedforward(obj, input)
            h = zeros(size(obj.W, 2), 1);
            chars = size(obj.corpus.allChars, 2);
            
            words  = obj.corpus.encodeString(input{:, 1})';
            x = RNN.one_hot(chars, words);
            tau = size(words, 2);
            
            a = obj.U * x(:, 1) + obj.W * zeros(size(obj.W, 2), 1) + obj.b; % h{0} = 0
            h = obj.activation_function(a);

            for t = 2 : tau
                a = obj.U * x(:, t) + obj.W * h + obj.b;
                h = obj.activation_function(a);
            end
            
            o = obj.V * h + obj.c;
            out = RNN.softmax(o, true);
        end
        
        function train(obj, numEpochs, batchSize, eta)
            % Initialize function variables
            chars = size(obj.corpus.allChars, 2);
            langs = size(obj.corpus.languages, 2);
            epoch = 1;
            obj.isLearning = true; % loop termination test

            % Initialize training, test, and validation sets
            shuffled_corpus = RNN.shuffle(obj.corpus.corpus);
            shuffled_corpus = shuffled_corpus(:, 1 : 50);
            [training_data, test_data, validation_data] = RNN.split_input(shuffled_corpus, 0.9, 0.05, 0.05);
            training_input = training_data(1, :);
            training_output = RNN.one_hot(langs, cell2mat(training_data(2, :)));
            test_input = test_data(1, :);
            test_output = RNN.one_hot(langs, cell2mat(test_data(2, :)));
            validation_input = validation_data(1, :);
            validation_output = RNN.one_hot(langs, cell2mat(validation_data(2, :)));
                        
            % print output header
            fprintf(RNN.report_header);
            
            % the main loop: train numEpochs times, or until perfect accuracy is reached:
            while obj.isLearning
                % an epoch consists of training the NN on every input/target given,
                % divided into mini-batches
                
                % Reset mini-batch loop
                mb_offset = 1;
                
                % update with SGD for each mini-batch
                while (mb_offset < size(training_input, 2))
                    % select a mini-batch from the inputs and targets
                    batchEnd = min(mb_offset + batchSize - 1, size(training_input, 2));
                    mb_inputs  = training_input(:, mb_offset : batchEnd);
                    mb_targets = training_output(:, mb_offset : batchEnd); % note that this is an array
                    mb_size = size(mb_inputs, 2);
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% feedforward
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    words  = obj.corpus.encodeString(mb_inputs{:, 1})';
                    x = RNN.one_hot(chars, words);
                    tau   = size(words, 2);
                    a     = cell(tau, 1);
                    h     = cell(tau, 1);
                    delta = cell(tau, 1);
                    
                    a{1} = obj.U * x(:, 1) + obj.W * zeros(size(obj.W, 2), 1) + obj.b; % h{0} = 0
                    h{1} = obj.activation_function(a{1});

                    for t = 2 : tau
                        a{t} = obj.U * x(:, t) + obj.W * h{t - 1} + obj.b;
                        h{t} = obj.activation_function(a{t});
                    end

                    o = obj.V * h{tau} + obj.c;
                    y_hat = RNN.softmax(o, true);
                                        
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% output error
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % the output error : ∂L = ∇_aC ⊙ σ′(z^L)
                    delta_o = obj.cost_function_prime(mb_targets, y_hat, o, @RNN.softmax_prime);
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% backpropagate the error
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % backprop through o
                    delta{tau} = obj.V' * delta_o .* RNN.softmax_prime(h{tau});
                        
                    % for each t = tau - 1, ... , 1 compute  δ^l = (w^{l+1}' * δ^{l+1}) ⊙ σ′(z^l)
                    % a{t} = obj.U * x + obj.W * h{t} + obj.b;

                    for t = tau - 1 : -1 : 1
                        delta{t} = ((obj.W' * delta{t+1}) + obj.U * x(:, t)) .* obj.activation_function_prime(a{t});
                    end
                    

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% update weights and biases
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % Delta_V, weights from hidden layer to output
                    Delta_V = eta * y_hat * delta{tau}'; % this assumes softmax
                    obj.V = obj.V - Delta_V;

                    % Delta_c, biases for output layer
                    Delta_c = eta * delta_o;
                    obj.c = obj.c - Delta_c;

                    % Delta_W, weights from hidden layer to hidden layer (through time)
                    % α( sum(δyj(t)y_l(t−1)))
                    Delta_W = 0;
                    for t = 2 : tau
                        Delta_W = Delta_W + eta * delta{t} * h{t - 1}';
                    end
                    obj.W = obj.W - Delta_W;

                    % Delta_U, weights from input to hidden layer
                    % α(sum ((T)δyj(t)xi(t)))
                    Delta_U = 0;
                    for t = 1 : tau
                        Delta_U = Delta_U + eta * delta{t} * x(:, t)';
                    end
                    obj.U = obj.U - Delta_U;

                    % Delta_b, biases for hidden layer
                    % scale * sum(delta{l}, 2);
                    Delta_b = 0;
                    for t = 1 : tau
                        Delta_b = Delta_b + eta * delta{t};
                    end
                    obj.b = obj.b - Delta_b;
                    
                    mb_offset = mb_offset + batchSize;
                end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%% output a report for each epoch
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % correct measures the number of matching columns between the given
                %   targets and the (rounded) outputs of the NN
                training_results = zeros(langs, size(training_input, 2));
                for i = 1 : size(training_input, 2)
                    training_results(:, i) = obj.feedforward(training_input(i));
                end
                training_size = size(training_results, 2);
                training_cost = obj.cost_function(training_output, training_results);
                training_correct = sum(all(training_output == round(training_results), 1));
                training_acc = training_correct / training_size;
                
                test_results = zeros(langs, size(test_input, 2));
                for i = 1 : size(test_input, 2)
                    test_results(:, i) = obj.feedforward(test_input(i));
                end
                test_size = size(test_results, 2);
                test_cost = obj.cost_function(test_output, test_results);
                test_correct = sum(all(test_output == round(test_results), 1));
                test_acc = test_correct / test_size;
                
                validation_results = zeros(langs, size(validation_input, 2));
                for i = 1 : size(validation_input, 2)
                    validation_results(:, i) = obj.feedforward(validation_input(i));
                end
                validation_size = size(validation_results, 2);
                validation_cost = obj.cost_function(validation_output, validation_results);
                validation_correct = sum(all(validation_output == round(validation_results), 1));
                validation_acc = validation_correct / validation_size;
                
                % store and output results
                obj.training_stats = [obj.training_stats ; training_acc training_cost test_acc test_cost validation_acc validation_cost];
                fprintf(1, RNN.report_format, epoch, ...
                    training_cost, training_correct, training_size, training_acc, ...
                    test_cost, test_correct, test_size, test_acc, ...
                    validation_cost, validation_correct, validation_size, validation_acc);
                
                % update loop and test for termination
                epoch = epoch + 1;
                if (epoch > numEpochs) || (test_correct == size(test_results, 2))
                    obj.isLearning = false;
                end
            end
        end
        
        function out = best_matches(obj, matches, n)
            % from a softmax array, returns a sorted list of the n
            % best-matching languages
            out = cell(1, 2);
            for i = 1 : size(matches, 1)
                out{i, 1} = matches(i);
                out{i, 2} = obj.corpus.languages{i};
            end
            out = RNN.max_n(out, n);
        end
    end
end
