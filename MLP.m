% coursework part 2
NINPUTS = 2;
NHIDDEN = 3;
NOUTPUT = 1;

Weights1 = zeros(NINPUTS,NHIDDEN);
Weights2 = zeros(NHIDDEN, NOUTPUT);
DirectWeights = zeros(NINPUTS,NOUTPUT);
HiddenBiases = zeros(NHIDDEN,1);
OutBias = zeros(NOUTPUT,1);
LearningRate = 0.2;

Weights1(1,1) = -0.1; Weights1(1,2) = -0.1; Weights1(1,3) = 0;
Weights1(2,1) = 0; Weights1(2,2) = 0.1; Weights1(2,3) = -0.3;

DirectWeights(1,1) = -0.3; DirectWeights(2,1) = 0.2;

Weights2(1,1) = 0.3; Weights2(2,1) = -0.1; Weights2(3,1) = 0.2;

HiddenBiases(1,1) = 0; HiddenBiases(2,1) = 0; HiddenBiases(3,1) = 0;
OutBias(1,1) = 0;

Inputs = [0 1 ; 1 0];
DesiredOutput = [1 ; 1];

for i = 1:NINPUTS

    fprintf('Starting Weights: w1=%.4f, w2=%.4f, w3=%.4f, w4=%.4f, w5=%.4f, w6=%.4f, w7=%.4f, w8=%.4f, w9=%.4f, w10=%.4f, w11=%.4f\n', ...
        Weights1(1,1),DirectWeights(1,1),Weights1(1,2),Weights1(2,2),DirectWeights(2,1),Weights1(2,3), Weights2(3,1), Weights2(2,1), Weights2(1,1), ...
        Weights1(1,3), Weights1(2,1))
    fprintf('Starting biases: b1=%.4f, b2=%.4f, b3=%.4f, bOut=%.4f\n', HiddenBiases(1,1), HiddenBiases(2,1), HiddenBiases(3,1), OutBias)

    input = Inputs(:, i);
    target = DesiredOutput(i);

    % Forward pass
    hidden_output = zeros(NHIDDEN, NOUTPUT);
    for j = 1:NHIDDEN
        for k = 1:NINPUTS
            hidden_output(j) = hidden_output(j) + input(k) * Weights1(k, j);
        end
        hidden_output(j) = hidden_output(j) + HiddenBiases(j);
        hidden_output(j) = sigma(hidden_output(j));
    end

    output_input = 0;
    for j = 1:NHIDDEN
        output_input = output_input + hidden_output(j) * Weights2(j);
    end
    for k = 1:NINPUTS
        output_input = output_input + input(k) * DirectWeights(k);
    end
    output = output_input + OutBias;

    % Back propagation
    beta_output = output * (1 - output) * (target - output);
    beta_hidden = zeros(NHIDDEN,NOUTPUT);
    for j = 1:NHIDDEN
        beta_hidden(j) = hidden_output(j) * (1 - hidden_output(j)) * (beta_output * Weights2(j));
    end

    for j = 1:NHIDDEN
        Weights2(j) = Weights2(j) + (LearningRate * beta_output * hidden_output(j));
    end

    for k = 1:NINPUTS
        DirectWeights(k) = DirectWeights(k) + (LearningRate * beta_output * input(k));
    end

    for k = 1:NINPUTS
        for j = 1:NHIDDEN
            Weights1(k,j) = Weights1(k,j) + (LearningRate * beta_hidden(j) * input(k));
        end
    end

    for j = 1:NHIDDEN
        HiddenBiases(j) = HiddenBiases(j) + (LearningRate * beta_hidden(j));
    end

    OutBias = OutBias + (LearningRate * beta_output);

    Weights1(1,3) = 0; Weights1(2,1) = 0;

    fprintf('Updated Weights: w1=%.4f, w2=%.4f, w3=%.4f, w4=%.4f, w5=%.4f, w6=%.4f, w7=%.4f, w8=%.4f, w9=%.4f, w10=%.4f, w11=%.4f\n', ...
        Weights1(1,1),DirectWeights(1,1),Weights1(1,2),Weights1(2,2),DirectWeights(2,1),Weights1(2,3), Weights2(3,1), Weights2(2,1), Weights2(1,1), ...
        Weights1(1,3), Weights1(2,1))
    fprintf('Updated biases: b1=%.4f, b2=%.4f, b3=%.4f, bOut=%.4f\n\n', HiddenBiases(1,1), HiddenBiases(2,1), HiddenBiases(3,1), OutBias)

end

% Sigma function
function sig = sigma(sum)
sig = 1.0./(1.0+exp(-sum));
sig = round(sig, 4);
end