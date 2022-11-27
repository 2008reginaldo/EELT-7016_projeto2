# Reginaldo Ferreira

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from collections import namedtuple

def convert(dictionary):
    ''' Converte um dicionário em uma namedTuple.
        Usefull
    '''
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def generateInput(size):
    '''
    input:
        size: tamanho do vetor u
    output:
        x: pandas Series. Vetor de saida u
    '''
    shift = 20
    # x = np.random.rand(size + shift)
    # x = np.random.choice([-1, 0, 1],  size+shift )
    x = np.random.choice([0, 1],  size+shift, p=[0.2, 0.8] )

    x = pd.Series(x)
    k = np.random.choice([2, 3, 4, 5])

    for i in range(1, k):
        x = x + pd.Series(x).shift(i)

    x = x[shift:]
    x.reset_index(inplace=True, drop=True)
    return x

def generateLags(x, nlags, delay, ZFMV):
    '''
    input:
        x: pandas Series > vetor de entrada
        nlags: int > numero de lags
        delay: int > delay
        ZFMV: bolean: Zero fill missing value
    output:
        matrix_dalay: pandas Dataframe > Matriz de dalays
    '''
    dic = {}

    fill_value = x[0] * (1-ZFMV)

    for i in range(nlags):
        # dic[i] = pd.Series(x).shift(i + delay, fill_value=0)
        dic[i] = x.shift(i + delay, fill_value=fill_value )

    matrix_dalay = pd.DataFrame(dic)
    return matrix_dalay

def generate_matrix(size, lags, delay, ZFMV):
    '''
    input:
        size: tamanho do vetor u
        lags: list > vetor coeficientes que mult os lags
        delay: int > delay
        ZFMV: bolean: Zero fill missing value
    output:
        u: pandas Series > Vetor de saida u
        matrix: pandas Dataframe > Matriz de lags * coeficientes dos lags
    '''
    u = generateInput(size)
    matrix = generateLags(u, len(lags), delay, ZFMV)
    matrix = matrix * lags

    return  u, matrix

def generate_error_matrix(size, lags, delay, ZFMV, std=1):
    '''
    input:
        size: tamanho do vetor u
        lags: list > vetor coeficientes que mult os lags
        delay: int > delay
        ZFMV: bolean > Zero fill missing value
        std: int > desvio padrão
    output:
        e: pandas Series > Vetor de saida e
        matrix: pandas Dataframe > Matriz de lags * coeficientes dos lags
    '''
    e = np.random.normal(0, std, size)
    e = pd.Series(e)
    matrix = generateLags(e, len(lags), delay, ZFMV )
    matrix = matrix * lags

    return  e, matrix

def generate_y(matrix, lags):
    n = len(lags)
    l = matrix.shape[0]
    y = [0.0] * n
    b =  np.array(lags)[::-1]

    for i in range(l):
        a = np.array(y)[-n:]
        yy = np.sum(a * -b) + np.sum(matrix[i])
        y.append(yy)
        # print('i:',i, 'a:', a, 'b', b, 'yy:', yy, 'y:', y)

    y = np.array(y)[n:]
    # y = np.array(y)
    y = pd.Series(y)

    return y

def generateDataset(model):
    m = convert(model)

    u, input_matrix = generate_matrix(m.size, m.B_coefs, m.delay, ZFMV=1)
    #u = m.input
    e, error_matrix =  generate_error_matrix(m.size, m.C_coefs, m.delay, ZFMV=1)
    matrix = np.concatenate((input_matrix, error_matrix), axis=1)
    y = generate_y(matrix, m.A_coefs)

    df = pd.DataFrame([u, y, e]).T
    df.columns = ['input', 'output', 'error']
    return df, matrix

def print_thetas(theta, n, m, use_b0):
    '''Printa os valores de Theta'''
    for i in range(m.A_order):
        print(f'a{i+1} = {theta[i]}')

    for j in range(1 - use_b0, m + 1):
        print(f'b{j} = {theta[j + n - 1]}')

# def make_phi(output, n, m, imput=None,  use_b0=True, ZFMV=True, delay=1):
#     '''
#     ZFMV: Zeros for missing values. If False, use the first value.
#     '''
#     if ZFMV:
#         fill = 0
#     else:
#         fill = 1
#
#     l = y[1:].shape[0]
#     phi_n = np.zeros(l * n).reshape(l, -1)
#     phi_m = np.zeros(l * m).reshape(l, -1)
#     y_fill = y[0] * fill
#     u_fill = u[0] * fill
#
#     for i in range(n):
#         phi_n[:, i] = np.append( np.zeros(i) + y_fill, y[:l-i] )
#
#     for j in range(m):
#         phi_m[:, j] = np.append( np.zeros(j) + u_fill, u[:l-j] )
#
#     if use_b0:
#         phi_m = np.append(u[1:l+1].reshape(l, -1), phi_m, axis=1)
#
#     phi =  np.append( -phi_n, phi_m, axis=1)
#
#     return phi


# def make_phi_arma(y, u, n, m, use_b0=True, ZFMV=True):
#     '''
#     ZFMV: Zeros for missing values. If False, use the first value.
#     '''
#     if ZFMV:
#         fill = 0
#     else:
#         fill = 1
#
#     l = y[1:].shape[0]
#     phi_n = np.zeros(l * n).reshape(l, -1)
#     phi_m = np.zeros(l * m).reshape(l, -1)
#     y_fill = y[0] * fill
#     u_fill = u[0] * fill
#
#     for i in range(n):
#         phi_n[:, i] = np.append( np.zeros(i) + y_fill, y[:l-i] )
#
#     for j in range(m):
#         phi_m[:, j] = np.append( np.zeros(j) + u_fill, u[:l-j] )
#
#     if use_b0:
#         phi_m = np.append(u[1:l+1].reshape(l, -1), phi_m, axis=1)
#
#     phi =  np.append( -phi_n, phi_m, axis=1)
#
#     return phi

def make_phi(y, u, n, m, use_b0=False, delay=0, ZFMV=True):
    '''
    ZFMV: Zeros for missing values. If False, use the first value.
    '''
    phi_n = generateLags(y, n, delay, ZFMV)
    phi_m = generateLags(u, m, delay, ZFMV)
    fill_value = u.values[-1:][0] * (1 - ZFMV)

    if use_b0:
        phi_b0 = u.shift( -1, fill_value=fill_value )
        phi =  pd.concat([phi_n, phi_b0, phi_m], axis=1)
    else:
        phi =  pd.concat([phi_n, phi_m], axis=1)
    return phi


def make_phi_armax(y, u, resi, n, m, c, use_b0=False, delay=0, ZFMV=True, use_c0=False):
    '''
    ZFMV: Zeros for missing values. If False, use the first value.
    '''
    phi_n = generateLags(y, n, delay, ZFMV)
    phi_m = generateLags(u, m, delay, ZFMV)
    phi_c = generateLags(resi, c, delay, ZFMV)
    fill_value = u.values[-1:][0] * (1 - ZFMV)

    if use_c0:
        phi_c0 = resi.shift( -1, fill_value=fill_value )
        phi_c =  pd.concat([phi_c0, phi_c], axis=1)

    if use_b0:
        phi_b0 = u.shift( -1, fill_value=fill_value )
        phi =  pd.concat([phi_n, phi_b0, phi_m, phi_c], axis=1)
    else:
        phi =  pd.concat([phi_n, phi_m, phi_c], axis=1)
    return phi

# def make_phi_arx(y, n, ZFMV=True):
#     '''
#     ZFMV: Zeros for missing values. If False, use the first value.
#     '''
#     if ZFMV:
#         fill = 0
#     else:
#         fill = 1
#
#     l = y[1:].shape[0]
#     phi_n = np.zeros(l * n).reshape(l, -1)
#     y_fill = y[0] * fill
#
#     for i in range(n):
#         phi_n[:, i] = np.append( np.zeros(i) + y_fill, y[:l-i] )
#
#     return phi_n


def MSE(y, y_predict):
    N = y.shape[0]
    J = np.sum( (y - y_predict)**2 )  /  N
    return J

def gerar_yest(model, theta, y, u, type, residue = None):

    m = convert(model)
    n = m.A_order
    indx = u.index

    if m.model == 'arx':
        print('MODELO ARX')
        phi = make_phi( y.reset_index(drop=True), u.reset_index(drop=True), m.A_order, m.B_order, m.use_b0, 0, m.ZFMV )
        phi = phi.values

        N = phi.shape[0]
        # print('N:', N)
        y_predict = np.zeros(N)
        for k in range(n):
            y_predict[k] = y.values[k]

        if type == 'yest_n':
            # print('n')
            for i in range(n, N):
                p1 = np.sum( -theta[:n] * np.flip(y_predict[i-n:i]) )
                p2 = np.sum( theta[n:] * phi[i, n:] )
                y_predict[i]  = p1 + p2

        if type == 'yest_1':
            # print('1')
            for i in range(n, N):
                p1 = np.sum( -theta[:n] * np.flip(y[i-n:i]) )
                p2 = np.sum( theta[n:] * phi[i, n:] )
                y_predict[i]  = p1 + p2

        mse = MSE(y, y_predict)
        y_predict = pd.Series(y_predict, index=indx)


    if m.model == 'armax':
        print('MODELO ARMAX')
        jj = m.B_order
        phi = make_phi_armax( y.reset_index(drop=True), u.reset_index(drop=True), residue.reset_index(drop=True), m.A_order, m.B_order, m.C_order, m.use_b0, 0, m.ZFMV )
        phi = phi.values

        N = phi.shape[0]
        # print('N:', N)
        y_predict = np.zeros(N)
        for k in range(n):
            y_predict[k] = y.values[k]

        if type == 'yest_n':
            # print('n')
            for i in range(n, N):
                p1 = np.sum( -theta[:n] * np.flip(y_predict[i-n:i]) )
                p2 = np.sum( theta[n:jj] * phi[i, n:jj] )
                p3 = np.sum( theta[jj:] * phi[i, jj:] )
                y_predict[i]  = p1 + p2 + p3

        if type == 'yest_1':
            # print('1')
            for i in range(n, N):
                p1 = np.sum( -theta[:n] * np.flip(y[i-n:i]) )
                p2 = np.sum( theta[n:jj] * phi[i, n:jj] )
                p3 = np.sum( theta[jj:] * phi[i, jj:] )
                y_predict[i]  = p1 + p2 + p3

        mse = MSE(y, y_predict)
        y_predict = pd.Series(y_predict, index=indx)

    return y_predict, mse

def gerar_yest_arx(model, theta, y, u, type, residue = None):
    m = convert(model)
    n = m.A_order
    indx = u.index

    print('MODELO ARX')
    phi = make_phi( y.reset_index(drop=True), u.reset_index(drop=True), m.A_order, m.B_order, m.use_b0, 0, m.ZFMV )
    phi = phi.values

    N = phi.shape[0]
    # print('N:', N)
    y_predict = np.zeros(N)
    for k in range(n):
        y_predict[k] = y.values[k]

    if type == 'yest_n':
        # print('n')
        for i in range(n, N):
            p1 = np.sum( -theta[:n] * np.flip(y_predict[i-n:i]) )
            p2 = np.sum( theta[n:] * phi[i, n:] )
            y_predict[i]  = p1 + p2

    if type == 'yest_1':
        # print('1')
        for i in range(n, N):
            p1 = np.sum( -theta[:n] * np.flip(y[i-n:i]) )
            p2 = np.sum( theta[n:] * phi[i, n:] )
            y_predict[i]  = p1 + p2

    mse = MSE(y, y_predict)
    y_predict = pd.Series(y_predict, index=indx)

    return y_predict, mse

def gerar_yest_armax(model, theta, y, u, type, residue = None):
    print('MODELO ARMAX')
    jj = m.B_order
    phi = make_phi_armax( y.reset_index(drop=True), u.reset_index(drop=True), residue.reset_index(drop=True), m.A_order, m.B_order, m.C_order, m.use_b0, 0, m.ZFMV )
    phi = phi.values

    N = phi.shape[0]
    # print('N:', N)
    y_predict = np.zeros(N)
    for k in range(n):
        y_predict[k] = y.values[k]

    if type == 'yest_n':
        # print('n')
        for i in range(n, N):
            p1 = np.sum( -theta[:n] * np.flip(y_predict[i-n:i]) )
            p2 = np.sum( theta[n:jj] * phi[i, n:jj] )
            p3 = np.sum( theta[jj:] * phi[i, jj:] )
            y_predict[i]  = p1 + p2 + p3

    if type == 'yest_1':
        # print('1')
        for i in range(n, N):
            p1 = np.sum( -theta[:n] * np.flip(y[i-n:i]) )
            p2 = np.sum( theta[n:jj] * phi[i, n:jj] )
            p3 = np.sum( theta[jj:] * phi[i, jj:] )
            y_predict[i]  = p1 + p2 + p3

    mse = MSE(y, y_predict)
    y_predict = pd.Series(y_predict, index=indx)

    return y_predict, mse


def gerar_yest_1(model, theta, y, u):
#
#     m = convert(model)
#     n = m.A_order
#     phi = make_phi_arma( y, u, m.A_order, m.B_order, m.use_b0, m.ZFMV )
#
#     N = phi.shape[0]
#     y_predict = np.zeros(N)
#     for k in range(n):
#         y_predict[k] = y[k]
#
#     mse = MSE(y[n+1:], y_predict[n:], n)
#
#     return mse, y_predict



def fit_arx(model, y, u):
    m = convert(model)
    Y = y[1:]
    delay = 0

    phi = make_phi( y, u, m.A_order, m.B_order, m.use_b0, delay, m.ZFMV)

    phi = phi.values[:-1]
    theta = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    theta[:m.A_order] *= -1
    return theta


def fit_armax(model, y, u, resi):
    m = convert(model)
    Y = y[1:]
    delay = 0

    phi = make_phi_armax( y, u, resi, m.A_order, m.B_order, m.C_order, m.use_b0, delay, m.ZFMV)
    phi = phi.values[:-1]
    theta = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    theta[:m.A_order] *= -1
    return theta


def fit(model, y, u):
    m = convert(model)

    if m.model == 'arx':
        theta = fit_arx(model, y, u)
    if m.model == 'armax':
        theta = fit_armax(model, y, u)
    return theta

def iterative_fit(model, y, u):



def ac(y, order=100):
#     '''
#     Esta funçao é para gerar os gráficos de autocorrelation
#     mas não chegei a utilizá-la no projeto
#     '''
#     ac_list = []
#     for i in range(1, order):
#         ac_ = np.corrcoef(y[i:], y[:-i])[0,1]
#         ac_list.append(ac_)
#
#     return np.array(ac_list)


def analise(model, data, r=0.7):
    '''
    Esta função separa od dados de entradas 'data' em conjuntos de validaçao
    e teste segundo à proporção r. O coeficientes e MSEs dos fittings do modelo
    'model' são registrados no dicionário result.
    '''
    n = int(data.shape[0] * r)

    y_train = data.output[:n]
    u_train = data.input[:n]
    theta = fit(model, y_train, u_train)
    yest_1_train, mse_1_train = gerar_yest(model, theta, y_train, u_train, 'yest_1')
    yest_n_train, mse_n_train = gerar_yest(model, theta, y_train, u_train, 'yest_n')
    residue_train_1 = y_train - yest_1_train
    residue_train_n = y_train - yest_n_train

    result = {
        'model': model,
        'theta': theta,

        'yest_1_train': yest_1_train,
        'yest_n_train': yest_n_train,
        'mse_1_train':  mse_1_train,
        'mse_n_train':  mse_n_train,

        'y_train': y_train,
        'u_train': u_train,

        'residue_train_1': residue_train_1,
        'residue_train_n': residue_train_n,
        }

    if r < 1:
        y_val = data.output[n:]
        u_val = data.input[n:]
        yest_1_val, mse_1_val = gerar_yest(model, theta, y_val, u_val, 'yest_1')
        yest_n_val, mse_n_val = gerar_yest(model, theta, y_val, u_val, 'yest_n')
        residue_val_1 = y_val - yest_1_val
        residue_val_n = y_val - yest_n_val

        result_part2 = {
            'yest_1_val': yest_1_val,
            'yest_n_val': yest_n_val,
            'mse_1_val': mse_1_val,
            'mse_n_val': mse_n_val,

            'y_val':   y_val,
            'u_val':   u_val,

            'residue_val_1': residue_val_1,
            'residue_val_n': residue_val_n,
        }

        result.update(result_part2)
    return result


def generate_error_vector(result_, type):
    if type == 'val':
        error = result_['yest_n_val'] - result_['y_val']
    if type == 'train':
        error = result_['yest_n_train'] - result_['y_train']
    return error


def analise2(model1, model2, data, r):
#
#     res = analise(model1, data, r)
#     error = generate_error_vector(res, 'train')
#     print('theta 1: ', res['theta'])
#     print('MSE_n_train 1: ', res['mse_n_train'])
#
#     data.input = error
#     data.output = res['yest_n_train']
#
#     res2 = analise(model2, data, r)
#     print('theta 2: ', res2['theta'])
#     print('MSE_n_train 2: ', res2['mse_n_train'])
#
#     return res, res2


# def RLS(model, data):
#     m = convert(model)
#     phi = make_phi()
#
#     pass


def summary(res_):

    ans = {}
    selected_keys = 'model theta mse_1_train mse_n_train mse_1_val mse_n_val'.split()
    for key in selected_keys:
        print(key,": ", res_[key])
        ans[key] = res_[key]
    residue_datas = 'residue_train_1, residue_train_n, residue_val_1, residue_val_n'.split(', ')
    for key in residue_datas:
        print(key,'.mean: ', res_[key].mean())
        ans[key + '.mean'] = res_[key].mean()
        print(key,'.std: ', res_[key].std())
        ans[key + '.std'] = res_[key].std()

    return ans


def comparacao_modelo(data):
    ans = {}
    for na in [1, 2, 3, 4]:
        for nb in [0, 1, 2, 3, 4]:
            for b0 in [0, 1]:
                for zfmv in [0, 1]:
                    fitModel = dict(A_order= na, B_order=nb, use_b0=b0, ZFMV=zfmv, model='arx')
                    ans[na, nb, b0, zfmv] = summary( analise(fitModel, data, 0.7) )
    return ans

