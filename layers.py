import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy(x):
    return -np.log(x + 1e-6)


def regularized_cross_entropy(layers, lam, x):
    loss = cross_entropy(x)
    for layer in layers:
        loss += lam * (np.linalg.norm(layer.get_weights()) ** 2)
    return loss


def leakyReLU(x, alpha=0.001):
    return x * alpha if x < 0 else x


def leakyReLU_derivative(x, alpha=0.01):
    return alpha if x < 0 else 1


def lr_schedule(learning_rate, iteration): 
    if iteration == 0:
        return learning_rate
    if (iteration >= 0) and (iteration <= 10000):
        return learning_rate
    if iteration > 10000:
        return learning_rate * 0.1
    if iteration > 30000:
        return learning_rate * 0.1

class Convolutional:                                  
    def __init__(self, name, num_filters=16, stride=1, size=3, activation=None):
        self.name = name
        self.filters = np.random.randn(num_filters, 3, 3) * 0.1
        self.stride = stride
        self.size = size
        self.activation = activation
        self.last_input = None
        self.leakyReLU = np.vectorize(leakyReLU)
        self.leakyReLU_derivative = np.vectorize(leakyReLU_derivative)

    def forward(self, image):
        self.last_input = image                            
        input_dimension = image.shape[1]                                            
        output_dimension = int((input_dimension - self.size) / self.stride) + 1       

        out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))    
                                                                                

        for f in range(self.filters.shape[0]):              
            tmp_y = out_y = 0                              
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = image[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    out[f, out_y, out_x] += np.sum(self.filters[f] * patch)
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        if self.activation == 'relu':                    
            self.leakyReLU(out)
        return out

    def backward(self, din, learn_rate=0.005):
        input_dimension = self.last_input.shape[1]         

        if self.activation == 'relu':                      
           self.leakyReLU_derivative(din)

        dout = np.zeros(self.last_input.shape)           
        dfilt = np.zeros(self.filters.shape)            

        for f in range(self.filters.shape[0]):            
            tmp_y = out_y = 0
            while tmp_y + self.size <= input_dimension:
                tmp_x = out_x = 0
                while tmp_x + self.size <= input_dimension:
                    patch = self.last_input[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]
                    dfilt[f] += np.sum(din[f, out_y, out_x] * patch, axis=0)
                    dout[:, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size] += din[f, out_y, out_x] * self.filters[f]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1
        self.filters -= learn_rate * dfilt                 
        return dout                                 

    def get_weights(self):
        return np.reshape(self.filters, -1)


class Pooling:                                           
    def __init__(self, name, stride=2, size=2):
        self.name = name
        self.last_input = None
        self.stride = stride
        self.size = size

    def forward(self, image):
        self.last_input = image                      

        num_channels, h_prev, w_prev = image.shape
        h = int((h_prev - self.size) / self.stride) + 1
        w = int((w_prev - self.size) / self.stride) + 1

        downsampled = np.zeros((num_channels, h, w))  

        for i in range(num_channels):                
            curr_y = out_y = 0                       
            while curr_y + self.size <= h_prev:     
                curr_x = out_x = 0
                while curr_x + self.size <= w_prev:   
                    patch = image[i, curr_y:curr_y + self.size, curr_x:curr_x + self.size]
                    downsampled[i, out_y, out_x] = np.max(patch) 
                    curr_x += self.stride                  
                    out_x += 1
                curr_y += self.stride
                out_y += 1

        return downsampled

    def backward(self, din, learning_rate):
        num_channels, orig_dim, *_ = self.last_input.shape      
        dout = np.zeros(self.last_input.shape)       
        for c in range(num_channels):
            tmp_y = out_y = 0
            while tmp_y + self.size <= orig_dim:
                tmp_x = out_x = 0
                while tmp_x + self.size <= orig_dim:
                    patch = self.last_input[c, tmp_y:tmp_y + self.size, tmp_x:tmp_x + self.size]   
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)        
                    dout[c, tmp_y + x, tmp_x + y] += din[c, out_y, out_x]
                    tmp_x += self.stride
                    out_x += 1
                tmp_y += self.stride
                out_y += 1

        return dout

    def get_weights(self):          
        return 0


class FullyConnected:            
    def __init__(self, name, nodes1, nodes2, activation):
        self.name = name
        self.weights = np.random.randn(nodes1, nodes2) * 0.1
        self.biases = np.zeros(nodes2)
        self.activation = activation
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None
        self.leakyReLU = np.vectorize(leakyReLU)
        self.leakyReLU_derivative = np.vectorize(leakyReLU_derivative)

    def forward(self, input):
        self.last_input_shape = input.shape    

        input = input.flatten()           

        output = np.dot(input, self.weights) + self.biases    

        if self.activation == 'relu':                   
            self.leakyReLU(output)

        self.last_input = input        
        self.last_output = output

        return output

    def backward(self, din, learning_rate=0.005):
        if self.activation == 'relu':                      
           self.leakyReLU_derivative(din)

        self.last_input = np.expand_dims(self.last_input, axis=1)
        din = np.expand_dims(din, axis=1)

        dw = np.dot(self.last_input, np.transpose(din))       
        db = np.sum(din, axis=1).reshape(self.biases.shape)    

        self.weights -= learning_rate * dw              
        self.biases -= learning_rate * db

        dout = np.dot(self.weights, din)
        return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)


class Dense:                   
    def __init__(self, name, nodes, num_classes):
        self.name = name
        self.weights = np.random.randn(nodes, num_classes) * 0.1
        self.biases = np.zeros(num_classes)
        self.last_input_shape = None
        self.last_input = None
        self.last_output = None

    def forward(self, input):
        self.last_input_shape = input.shape  

        input = input.flatten()           

        output = np.dot(input, self.weights) + self.biases  

        self.last_input = input    
        self.last_output = output

        return softmax(output)

    def backward(self, din, learning_rate=0.005):
        for i, gradient in enumerate(din):
            if gradient == 0:                  
                continue   
            t_exp = np.exp(self.last_output)        
            dout_dt = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            dout_dt[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)

            dt = gradient * dout_dt        

            dout = self.weights @ dt                              

            # update weights and biases
            self.weights -= learning_rate * (np.transpose(self.last_input[np.newaxis]) @ dt[np.newaxis])
            self.biases -= learning_rate * dt

            return dout.reshape(self.last_input_shape)

    def get_weights(self):
        return np.reshape(self.weights, -1)