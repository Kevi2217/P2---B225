import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from math import sqrt , atan , sin , cos
from numpy.linalg import inv

def read_csv():
    base_dir = Path('C:\\Users\\MikkelNørgaard\\Downloads\\')
    file = base_dir / 'DSTNationalregnskab.csv'
    df = pd.read_csv(file)
    return df

def calculate_mean_value(df, df_column, series_row_start, series_row_end):
    series = df[df_column]
    series = series [series_row_start - 1994 : series_row_end - 1994 + 1]    
    mean = series.mean()
    return mean

def difference_equation_solution (t, b, k, y_1, y_2, G_start):
    Y0 = np.matrix([[y_1], [y_2]])
    A = np.matrix([[0, 1 ], [-k * b , b * k + b]])
    I2 = np.matrix([[1, 0], [0, 1]])
    G_matrix = np.matrix([[0],[G_start]])
    Yt = A ** t * Y0 + (A ** t - I2) * (A - I2) ** - 1 * G_matrix
    return Yt

def createplot (t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data):
    x =[]
    y =[]
    t_index = range (t_start - 1994, t_end - 1994 + 1)
    for i, t in enumerate(t_index):
        print(i)
        x.append(t + 1994)
        y.append(difference_equation_solution(i, b, k, y_1, y_2, G_start) [0, 0])
    plt.style.use('seaborn-poster')
    plt.plot(x, y, label = 'Beregnet Indenlandsk BNP', zorder = 2)
    plt.ylabel('Y(t) i mia. kr.')
    plt.xlabel('t')
    plt.plot(x_data , y_data , label = 'Faktisk Indenlandsk BNP', zorder = 1)
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlim(left = 1994)
    plt.show()

def calculate_square (t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data) :
    y = []
    square_sum = 0
    t_index = range ( t_start - 1994 , t_end - 1994 +1)
    for i, t in enumerate(t_index):
            y.append (difference_equation_solution (i, b, k, y_1, y_2, G_start ) [0, 0])
    for i, t in enumerate(t_index):
        difference = y[i] - y_data[i + (t_start - 1994)]
        square_sum = square_sum + difference ** 2
    return (square_sum)

def least_squares (t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data):
    least_squares = float ( 'inf' )
    for b1 in range (0, 1000):
        print(b1)
        b = b1 / 1000
        k = 1
        for k1 in range (0, 100):
            k = (k1 + 0.1) / 10
            if b * k < 1:
                if ((b * k + b) ** 2) - 4 * k * b != 0:
                    square = calculate_square (t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data)
                    if square < least_squares :
                        least_squares = square
                        b_best = b
                        k_best = k
    return (least_squares, b_best, k_best)

def main():
    df = read_csv()
    t_start = 2003
    t_end = 2006
    b, k = 0.507, 1.676
    y = df['Y (Indenlandsk)']
    y_1 = y[t_start - 1994] # 
    y_2 = y[t_start - 1994  + 1] # 
    G = df['G']
    G_start = calculate_mean_value(df, 'G', t_start, t_end)  #G[t_start - 1994]
    print(G_start)
    x_data = df['t']
    y_data = y

    #print(calculate_mean_value(df, 'B', 2003, 2006))
    #print(calculate_mean_value(df, 'K', 2003, 2006))

    #createplot(t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data)

    print(least_squares(t_start, t_end, b, k, y_1, y_2, G_start, x_data, y_data))
    
    
    # (28350.46365949355, 0.779, 0.01) 1995 to 2006
    
    # (1.6013236472161285, 0.777, 0.01) 1995 to 1998
    # (42.55367325024625, 0.742, 0.01) 1999 to 2002
    # (351.0692632894576, 0.738, 1.31) 2003 to 2006


    #print('1995 to 2006 converges towards', difference_equation_solution (100000000000, 0.779, 0.01, y[1995 - 1994], y[1995 - 1994  + 1], G[1995 - 1994]))

    #print('1995 to 1999 converges towards', difference_equation_solution (100000000000, 0.777, 0.01, y[1995 - 1994], y[1995 - 1994  + 1], G[1995 - 1994]))
    #print('1999 to 2002 converges towards', difference_equation_solution (100000000000, 0.742, 0.01, y[1999 - 1994], y[1999 - 1994  + 1], G[1999 - 1994]))
    #print('2003 to 2006 converges towards', difference_equation_solution (100000000000, 0.738, 1.31, y[2003 - 1994], y[2003 - 1994  + 1], G[2003 - 1994]))


if __name__ == '__main__':
	main()
