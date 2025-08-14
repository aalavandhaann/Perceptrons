import math

GROUND_TRUTH_AND: list[list[float]] = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
]

GROUND_TRUTH_OR: list[list[float]] = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

GROUND_TRUTH_ADD: list[list[float]] = [
    [0, 0, 0],
    [1, 0, 1],
    [1, 2, 3],
    [0, 2, 2],
    [1, 3, 4],
    [2, 3, 5]
]

def weightedSum(data: list[float], weights: list[float], bias: float):
    ws: float = 0.0
    for i, value in enumerate(data):
        weight: float = weights[i]
        ws += weight * value
    
    ws += bias
    return ws

def updateWeights(data: list[float], weights: list[float], error: float, learning_rate: float)->list[float]:
    new_weights: list[float] = []
    for i, d in enumerate(data):
        weight: float = weights[i]
        new_weight: float = weight + ( learning_rate * error * d)
        new_weights.append(new_weight)

    return new_weights

def updateBias(currentBias: float, learning_rate: float, error: float):
    return currentBias + (learning_rate * error)

def step(z: float)->float:
    if(z >= 0.5):
        return 1.0
    return 0.0

def sigmoid(z: float)->float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def activation(z: float, type: str='step'):
    if(type == 'sigmoid'):
        return sigmoid(z)
    elif (type == 'both'):
        p = sigmoid(z)
        return step(p)
    return step(z)
    

def predict(x1:float, x2: float, weights: list[float], bias: float):
    z = weightedSum([x1, x2], weights, bias)
    y_hat = activation(z, type='sigmoid')
    print(f'The result for x1: {x1} && x2: {x2} is: {int(y_hat)}')

def train(ground_truth: list[list[float]], weights: list[float], bias: float = 0.0, learning_rate: float = 0):
    
    converged: bool = False
    epochs: int = 0

    while(not converged):
        print(f'EPOCH :: {epochs + 1}')
        epoch_is_updated: bool = False
        epoch_updates: int = 0
        for sample in ground_truth:
            x1, x2, y = sample
            z: float = weightedSum([x1, x2], weights, bias)
            y_hat: float = activation(z, type='sigmoid')
            error: float = y - y_hat
            new_weights = updateWeights([x1, x2], weights, error, learning_rate)
            new_bias = updateBias(bias, learning_rate, error)

            nw1, nw2 = new_weights
            w1, w2 = weights

            epoch_is_updated = (nw1 != w1) or (nw2 != w2) or (new_bias != bias)
            weights = new_weights
            bias = new_bias

            if(epoch_is_updated):
                epoch_updates += 1
            # print(z, y_hat, error, weights, bias)
        
        epochs += 1

        if(epoch_updates == 0 or epochs > 200):
            print(f'BREAKING AT EPOCH: {epochs+1} with weights: {weights}, bias: {bias}')
            break
    
    return weights, bias


if __name__ == '__main__':
    weights, bias = train(GROUND_TRUTH_ADD, [0, 0], 0.0, 1.0)

    while(True):
        x1: float = float(input('Enter the value for x1: '))
        x2: float = float(input('Enter the value for x2: '))
        predict(x1, x2, weights, bias)
        exit: int = int(input('Do you want to exit? '))
        if(exit == 1):
            break 

