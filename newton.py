import numpy as np

def derivative(x, fun):
    result = (fun(start + 1e-6) - fun(start))/1e-6
    return result

def second_derivative(x, fun):
    result = (fun(start + 1e-6) - 2 * fun(start) + fun(start - 1e-6)) / ((1e-6) ** 2)
    return result

def optimize(start, fun):
    x_old = start
    x_new = x_old - derivative(x_old, fun) / second_derivative(x_old, fun)
    while(abs(x_new - x_old) > 1e-6):
        temp = x_new
        x_new = x_old - derivative(x_old, fun) / second_derivative(x_old, fun)
        x_old = temp
    return x_new