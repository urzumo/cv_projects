import torch
import numpy as np
import matplotlib.pyplot as plt

# по тензору далее будут производные, поэтому requires_grad=True
x = torch.tensor(
    [8., 8.], requires_grad=True)

# зададим оптимайзер и lr
learning_rate = 3e-4
optimizer = torch.optim.SGD([x], lr=learning_rate)

# определим функцию потерь
def loss_func(variable):
    return 10 * (variable ** 2).sum()

# функция SGD
def make_gradient_step(function, variable, optimizer):
    optimizer = optimizer
    function_result = function(variable)
    function_result.backward()
    optimizer.step()
    optimizer.zero_grad()
    
# подготовим переменные для визуализации
var_history = []
fn_history = []

# собственно SGD с данными для визуализации
for i in range(500):
    var_history.append(x.data.cpu().numpy().copy())
    fn_history.append(loss_func(x).data.cpu().numpy().copy())
    make_gradient_step(loss_func, x, optimizer)
    
# функция визуализации
def show_contours(objective,
                  x_lims=[-10.0, 10.0], 
                  y_lims=[-10.0, 10.0],
                  x_ticks=100,
                  y_ticks=100):
    x_step = (x_lims[1] - x_lims[0]) / x_ticks
    y_step = (y_lims[1] - y_lims[0]) / y_ticks
    X, Y = np.mgrid[x_lims[0]:x_lims[1]:x_step, y_lims[0]:y_lims[1]:y_step]
    res = []
    for x_index in range(X.shape[0]):
        res.append([])
        for y_index in range(X.shape[1]):
            x_val = X[x_index, y_index]
            y_val = Y[x_index, y_index]
            res[-1].append(objective(np.array([[x_val, y_val]]).T))
    res = np.array(res)
    plt.figure(figsize=(7,7))
    plt.contour(X, Y, res, 100)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
# визуализируем градиентный спуск
show_contours(loss_func)
plt.scatter(np.array(var_history)[:,0], np.array(var_history)[:,1], s=10, c='r');

# скорость оптимизации функции потерь
plt.figure(figsize=(7,7))
plt.plot(fn_history);
plt.xlabel('step')
plt.ylabel('function value');
