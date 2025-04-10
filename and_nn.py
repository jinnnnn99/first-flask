import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

class ANDNeuralNetwork:
    def __init__(self):
        self.weights_input_hidden = np.array([[1, 1], [1, 1]]) # 2x2の重み行列
        self.bias_hidden = np.array([-1.5, -1.5])              # 中間層のbias
        self.weights_hidden_output = np.array([1, 1])          # 2x1の重み行列
        self.bias_output = -0.5                                # 出力層のbias
    
    def forward(self, x):
        # 中間層の計算
        hidden_input = np.dot(x, self.weights_input_hidden.T) + self.bias_hidden
        # step関数以外の活性化関数にする場合、コメントアウトを切り替える
        hidden_output = step_function(hidden_input)
        #hidden_output = sigmoid(hidden_input)
        #hidden_output = relu(hidden_input)
        #hidden_output = tanh(hidden_input)
        
        # 出力層の計算
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        # step関数以外の活性化関数にする場合、コメントアウトを切り替える
        final_output = step_function(final_input)
        #final_output = sigmoid(final_input)
        #final_output = relu(final_input)
        #final_output = tanh(final_input)
        #final_output = softmax(final_input)
        #final_output = np.where(final_output >= 0.5, 1, 0)     # 2値化

        return final_output

if __name__ == "__main__":
    # ANDの入力と出力
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_outputs = np.array([0, 0, 0, 1])

    and_nn = ANDNeuralNetwork()

    for i, input in enumerate(inputs):
        output = and_nn.forward(input)
        print(f"Input: {input}, Expected Output: {expected_outputs[i]}, NN Output: {output}")
