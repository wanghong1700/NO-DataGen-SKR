#!/usr/bin/env python
# -*- coding:utf-8 _*-
# 此test文件测试petsc4py数据封装成dat文件
# 使用ctypes调用c函数

import numpy as np
import scipy
from scipy.interpolate import interp2d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, gmres
from scipy.io import loadmat

import torch
import matplotlib.pyplot as plt
import math

from scipy.fftpack import idct

from petsc4py import PETSc

import ctypes
import os
import subprocess
import stat
import pickle
from datetime import datetime
import time

from sklearn.decomposition import PCA


def GRF2(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.randn(s, s)

    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = (tau ** (alpha - 1) * (np.pi ** 2 * (K1 ** 2 + K2 ** 2) + tau ** 2) ** (-alpha / 2))

    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0

    # Inverse Discrete Cosine Transform (IDCT)
    U = idct(idct(L, norm='ortho').T, norm='ortho').T

    return U


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                    (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


### 这个函数实现的darcy flow考虑了a的变化
def solve_gwf(coef, F):
    K = coef.shape[0]

    X1, Y1 = np.meshgrid(np.arange(1 / (2 * K), 1, 1 / K), np.arange(1 / (2 * K), 1, 1 / K))
    X2, Y2 = np.meshgrid(np.arange(0, 1, 1 / (K - 1)), np.arange(0, 1, 1 / (K - 1)))

    interp_func = interp2d(X1, Y1, coef, kind='linear')
    coef = interp_func(X2[0, :], Y2[:, 0])

    interp_func_F = interp2d(X1, Y1, F, kind='linear')
    F = interp_func_F(X2[0, :], Y2[:, 0])

    F = F[1:K - 1, 1:K - 1]

    d = [[None] * (K - 2) for _ in range(K - 2)]

    for j in range(1, K - 1):
        diag_values = np.array([
            -0.5 * (coef[1:K - 2, j] + coef[2:K - 1, j]),
            0.5 * (coef[0:K - 2, j] + coef[1:K - 1, j]) + 0.5 * (coef[2:K, j] + coef[1:K - 1, j]) + \
            0.5 * (coef[1:K - 1, j - 1] + coef[1:K - 1, j]) + 0.5 * (coef[1:K - 1, j + 1] + coef[1:K - 1, j]),
            np.concatenate(([0], -0.5 * (coef[1:K - 2, j] + coef[2:K - 1, j])))
        ])

        d[j - 1][j - 1] = diags(diag_values, [-1, 0, 1], (K - 2, K - 2))

        if j != K - 2:
            off_diag = diags(-0.5 * (coef[1:K - 1, j] + coef[1:K - 1, j + 1]), 0, (K - 2, K - 2))
            d[j - 1][j] = off_diag
            d[j][j - 1] = off_diag
        print(j)
    A = np.bmat(d) * (K - 1) ** 2
    P = np.zeros((K, K))
    P[1:K - 1, 1:K - 1] = np.reshape(np.linalg.solve(A.todense(), F.ravel()), (K - 2, K - 2))

    interp_func = interp2d(X2, Y2, P, kind='cubic')
    P = interp_func(X1[0, :], Y1[:, 0])

    return P.T


### 这个函数的实现把系数a提了出来
def solve_darcy1(coef, f):
    # 定义问题参数
    s = coef.shape[0] - 2  # 网格尺寸
    h = 1.0 / (s - 1)  # 网格步长

    f = f[1:-1, 1:-1]
    coef_pos = coef[1:-1, 2:]
    coef_pos[:, -1] = 0
    coef_neg = coef[1:-1, :-2]
    coef_neg[:, 0] = 0
    # 计算系数矩阵
    # coef_flat = coef.flatten()
    diagonals = [-coef_neg.flatten()[1:], -coef[:-2, 1:-1].flatten()[s - 2:], 4 * coef[1:-1, 1:-1].flatten(),
                 -coef[2:, 1:-1].flatten()[:-(s - 2)], -coef_pos.flatten()[:-1]]
    offsets = [-1, -s, 0, s, 1]
    A = diags(diagonals, offsets, shape=(s * s, s * s))

    ## removes

    # 创建向量b
    b = h ** 2 * f.flatten()

    # 使用scipy的稀疏线性系统求解器求解
    # u, exitcode = gmres(A, b)
    u = spsolve(A, b)
    print(np.allclose(A.dot(u), b))

    # 将解向量重新整形为网格形式
    u = u.reshape((s, s))

    return u


def build_darcy(coef, f):
    s = coef.shape[0] - 2  # 网格尺寸
    h = 1.0 / (s - 1)  # 网格步长

    f = f[1:-1, 1:-1]
    coef_pos = coef[1:-1, 2:]
    coef_pos[:, -1] = 0
    coef_neg = coef[1:-1, :-2]
    coef_neg[:, 0] = 0
    # 计算系数矩阵
    # coef_flat = coef.flatten()
    diagonals = [-coef_neg.flatten()[1:], -coef[:-2, 1:-1].flatten()[s - 2:], 4 * coef[1:-1, 1:-1].flatten(),
                 -coef[2:, 1:-1].flatten()[:-(s - 2)], -coef_pos.flatten()[:-1]]
    offsets = [-1, -s, 0, s, 1]
    A = diags(diagonals, offsets, shape=(s * s, s * s))

    ## removes

    # 创建向量b
    b = h ** 2 * f.flatten()
    return A, b


def build_folder(exp_id, dataset, solver_array, precond_array, tol_array, max_iter, size_mat, num_data):
    # 建立这个每次运算所需的文件夹

    # 获取当前日期
    # current_date = datetime.now().strftime('%Y%m%d')  # YYYYMMDD format
    current_date = exp_id

    # 建立data文件夹 下面这些文件都是在这个文件夹下面
    # 检查文件夹是否存在
    if not os.path.exists('./data'):
        # 创建文件夹
        os.makedirs('./data')

    # 针对这次运算的参数在data下建立一个文件夹，下面的数据都是在这个文件夹下
    # 定义相对路径
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, current_date)

    # 检查文件夹是否存在
    if not os.path.exists(rel_path):
        # 创建文件夹
        os.makedirs(rel_path)

    # 建立文件夹 data_数据集_矩阵大小_数量 存放python直接可以读的数据文件 未排序
    # 检查文件夹是否存在
    if not os.path.exists(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data)):
        # 创建文件夹
        os.makedirs(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data))

    # 检查文件夹是否存在
    if not os.path.exists(rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data)):
        # 创建文件夹
        os.makedirs(rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data))

    # 建立文件夹 data_数据集_算法_预处理_误差_最大迭代次数_矩阵大小_数量
    # 存放所有的二进制的 A，x，b 用于c版本的petsc程序调用 已经排好序了
    # 检查文件夹是否存在
    for solver in solver_array:
        for precond in precond_array:
            for tol in tol_array:
                if not os.path.exists(
                        rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                       size_mat,
                                                                       num_data)):
                    # 创建文件夹
                    os.makedirs(
                        rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                       size_mat,
                                                                       num_data))

    # 建立文件夹 output
    # 检查文件夹是否存在
    if not os.path.exists(rel_path + '/output'):
        # 创建文件夹
        os.makedirs(rel_path + '/output')

    # 在output里面建立文件夹 output_数据集_算法_预处理_误差_最大迭代次数_矩阵大小_数量
    # 存放这次实验得到的参数 里面应该有 每次运算的迭代次数 误差 运算时间
    # 一个总和文件夹 total 存放 所有的迭代次数、误差、运算时间 以及一些分析文件 一些图表
    for solver in solver_array:
        for precond in precond_array:
            for tol in tol_array:
                if not os.path.exists(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                                size_mat,
                                                                                num_data)):
                    # 创建文件夹
                    os.makedirs(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                                size_mat,
                                                                                num_data))
                if not os.path.exists(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol,
                                                                                      max_iter,
                                                                                      size_mat,
                                                                                      num_data)):
                    # 创建文件夹
                    os.makedirs(
                        rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol,
                                                                                      max_iter,
                                                                                      size_mat,
                                                                                      num_data))

    # 建立文件夹seq 存放python原data和c版本二进制data的对应关系
    # 计算序列、区域划分、每个区域的信息以及矩阵属于哪些区域 一个class数组 存放每个矩阵对应的坐标 一个坐标数组
    # 检查文件夹是否存在
    if not os.path.exists(rel_path + '/seq'):
        # 创建文件夹
        os.makedirs(rel_path + '/seq')

    if not os.path.exists(rel_path + '/results'):
        # 创建文件夹
        os.makedirs(rel_path + '/results')

    return rel_path


def normalize_array(arr):
    # 获取数组的形状
    num_elements, num_dimensions = arr.shape

    # 找到每个维度的最小值和最大值
    min_values = np.min(arr, axis=0)
    max_values = np.max(arr, axis=0)

    # 归一化数组
    normalized_arr = (arr - min_values) / (max_values - min_values)

    return normalized_arr


def decimal_to_fixed_size_binary(decimal_num, size=10):
    binary_str = bin(decimal_num)[2:]  # Convert decimal to binary string and remove '0b' prefix

    # Add leading zeros to achieve the fixed size
    if len(binary_str) < size:
        binary_str = '0' * (size - len(binary_str)) + binary_str
    else:
        binary_str = binary_str[-size:]

    binary_array = [int(bit) for bit in binary_str]  # Convert each bit to an integer and store in an array
    return binary_array


def binary_sequence(n):
    sequence = []
    current = [0] * n

    def generate_numbers(current, index):
        if index == n:
            sequence.append(current[:])
            return

        generate_numbers(current, index + 1)
        current[index] = 1 - current[index]
        generate_numbers(current, index + 1)

    generate_numbers(current, 0)
    return sequence


def binary_array_to_decimal(binary_array):
    binary_str = ''.join(str(bit) for bit in binary_array)  # Convert binary array to a string
    decimal_num = int(binary_str, 2)  # Convert binary string to decimal
    return decimal_num


def convert_binary_arrays_to_decimal(main_array):
    decimal_array = [binary_array_to_decimal(binary_arr) for binary_arr in main_array]
    return decimal_array


class area:
    def __init__(self, area_dim, area_self, area_up=0, area_down=0, area_scope=[], mat_num=[]):
        self.area_dim = area_dim
        self.area_self = area_self
        self.area_up = area_up
        self.area_down = area_down
        self.area_scope = area_scope
        self.mat_num = mat_num


def dataset_generator(dataset, size_mat, num_data, rel_path, prm_freq, dim_pca):
    # 生成数据集存入 data_数据集_矩阵大小_数量
    # 将对应的线性方程组提取参数 制作成坐标数组 存入seq文件夹

    # 生成数据集
    if dataset == 'possion2d' or dataset == 'possion2d_mpi':
        # 生成数据集
        s = int(np.sqrt(size_mat))
        GRF = GaussianRF(2, s, alpha=2.5, tau=7, device='cpu')
        f = np.ones([s, s])
        # 随机独立生成
        w = np.exp(GRF.sample(num_data))

        # 生成线性方程组A和b数组（每一对元素都是一个线性方程组）
        A = []
        b = np.zeros([num_data, (s - 2) * (s - 2)])

        for i in range(num_data):
            A0, b[i] = build_darcy(w[i], f)
            A.append(A0)

    # elif dataset == 'x':

    np.save(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data) + '/A.npy', A)
    np.save(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data) + '/b.npy', b)
    w_np = w.numpy()
    np.save(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data) + '/w.npy', w_np)
    return A, b, w

def comp_frobenius_distance(A, B):
    diff = A - B
    return np.sqrt(np.trace(diff.conj().T @ diff))

def seq_generator(dataset, size_mat, num_data, rel_path, prm_freq, dim_pca, w):
    start_time = time.perf_counter()
    # # 生成坐标数组
    # # 对张量中的每个矩阵做傅立叶变换
    # fft_w = torch.fft.fft2(w)
    # # 使用 fftshift 将低频部分移到频谱的中心
    # fft_shift_tensor = torch.fft.fftshift(fft_w)
    # n = fft_shift_tensor.shape[-1]  # tensor 的维度
    # m = prm_freq
    # # 计算频谱中心的索引
    # center_index = n // 2
    # # 计算要取出的部分的起始和结束索引
    # start = center_index - m // 2
    # end = center_index + m // 2
    # # 取出中心的 m*m 部分
    # low_freq_part = fft_shift_tensor[:, start:end, start:end]
    # # 使用reshape将每个mxm矩阵转化为m^2维向量
    # low_freq_vectors = low_freq_part.reshape(num_data, -1)
    # # 将复数数组分解为实部和虚部
    # real_part = low_freq_vectors.real
    # imag_part = low_freq_vectors.imag
    # # 将实部和虚部连接在一起形成一个1D数组
    # concatenated_array = np.concatenate((real_part, imag_part), axis=1)
    # # 创建PCA对象，设置目标维度为dim_pca
    # pca = PCA(n_components=dim_pca)
    # # 对数据进行PCA降维
    # low_freq_pca = pca.fit_transform(concatenated_array)
    # address = normalize_array(low_freq_pca)
    # np.save(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data) + '/address.npy', address)

    address = []
    area_array = []
    area_num_totel = 0
    area_seq = []
    mat_seq = []

    lst = [i for i in range(1, num_data)]
    mat_seq.append(0)

    i = 0
    for i0 in range(num_data - 1):
        dis = 10000
        j = 0
        for j0 in lst:
            dis0 = comp_frobenius_distance(w[i], w[j0])
            # dis0 = comp_frobenius_distance(low_freq_part[i], low_freq_part[j0])
            # dis0 = calculate_distance(address[i], address[j0])
            if dis0 < dis:
                dis = dis0
                j = j0
        i = j
        mat_seq.append(j)
        lst.remove(j)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    with open(rel_path + '/results/seq_generate_time.txt', 'w') as file:
        file.write(str(total_time))
        file.close()

    return address, area_array, area_num_totel, area_seq, mat_seq


def seq_generator_vec(dataset, size_mat, num_data, rel_path, prm_freq, dim_pca, w):
    start_time = time.perf_counter()
    # 向量版本的排序算法
    # 生成坐标数组
    # 对张量中的每个矩阵做傅立叶变换
    fft_results = torch.fft.fft(w, n=w.shape[1])
    # 选择每个向量的低频m个元素
    m = prm_freq
    low_freq_vectors = fft_results[:, :m]
    # 将复数数组分解为实部和虚部
    real_part = low_freq_vectors.real
    imag_part = low_freq_vectors.imag
    # 将实部和虚部连接在一起形成一个1D数组
    concatenated_array = np.concatenate((real_part, imag_part), axis=1)
    # 创建PCA对象，设置目标维度为dim_pca
    pca = PCA(n_components=dim_pca)
    # 对数据进行PCA降维
    low_freq_pca = pca.fit_transform(concatenated_array)
    address = normalize_array(low_freq_pca)
    np.save(rel_path + '/data_{}_{}_{}'.format(dataset, size_mat, num_data) + '/address.npy', address)

    area_array = []
    for i in range(2 ** dim_pca):
        area0 = area(dim_pca, i)
        fixed_size_binary_array = decimal_to_fixed_size_binary(i, size=dim_pca)
        for j in range(dim_pca):
            if fixed_size_binary_array[j] == 0:
                area0.area_scope.append([0, 1 / 2])
            if fixed_size_binary_array[j] == 1:
                area0.area_scope.append([1 / 2, 1])
        area_array.append(area0)

    area_num_totel = 2 ** dim_pca
    sequence = binary_sequence(dim_pca)
    area_seq = convert_binary_arrays_to_decimal(sequence)

    for i in range(2 ** dim_pca):
        if i == 0:
            area_array[i].area_up = 0
            area_array[i].area_down = area_seq[i + 1]
        if i == 2 ** dim_pca - 1:
            area_array[i].area_up = area_seq[i - 1]
            area_array[i].area_down = 0
        else:
            area_array[i].area_up = area_seq[i - 1]
            area_array[i].area_down = area_seq[i + 1]

    for i in range(num_data):
        mat_address = []
        for j in range(dim_pca):
            if address[i][j] < 1 / 2:
                mat_address.append(0)
            else:
                mat_address.append(1)
        area_num = binary_array_to_decimal(mat_address[::-1])
        area_array[area_num].mat_num.append(i)

    # 这里缺一个自适应划分区域的模块

    mat_seq = []
    area_num0 = 0

    for i in range(area_num_totel):
        area_down = area_array[area_num0].area_down
        mat_num = area_array[area_num0].mat_num
        mat_seq.extend(mat_num)
        if area_down == 0:
            break
        else:
            area_num0 = area_down

    end_time = time.perf_counter()
    total_time = end_time - start_time
    with open(rel_path + '/results/seq_generate_time.txt', 'w') as file:
        file.write(str(total_time))
        file.close()

    return address, area_array, area_num_totel, area_seq, mat_seq


def petsc_generator(dataset, size_mat, num_data, rel_path, A, b, mat_seq):
    s = int(np.sqrt(size_mat))
    # 将线性方程组转换为PETSc格式然后输出dat格式
    for i in range(num_data):
        A0 = A[mat_seq[i]]
        b0 = b[mat_seq[i]]

        # 创建一个PETSc矩阵
        A_petsc = PETSc.Mat()
        A_petsc.create(PETSc.COMM_WORLD)
        A_petsc.setSizes([(s - 2) * (s - 2), (s - 2) * (s - 2)])
        A_petsc.setType('aij')  # 稀疏矩阵
        A_petsc.setUp()

        # 创建一个PETSc向量
        b_petsc = PETSc.Vec()
        b_petsc.create(PETSc.COMM_WORLD)
        b_petsc.setSizes((s - 2) * (s - 2))
        b_petsc.setUp()

        # 进行一些操作填充矩阵和向量...
        A_csr = A0.tocsr()
        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        b_petsc = PETSc.Vec().createWithArray(b0, comm=PETSc.COMM_WORLD)

        # 为了保存这些对象，我们需要为每个对象创建一个Viewer

        viewer_A = PETSc.Viewer().createBinary(
            rel_path + '/data_{}_{}_{}_PETSc/'.format(dataset, size_mat, num_data) + 'A_%d.dat' % i, 'w')
        viewer_b = PETSc.Viewer().createBinary(
            rel_path + '/data_{}_{}_{}_PETSc/'.format(dataset, size_mat, num_data) + 'rhs_%d.dat' % i, 'w')

        # 使用Viewers将矩阵和向量分别写入不同的二进制文件
        A_petsc.view(viewer_A)
        b_petsc.view(viewer_b)

        viewer_A.destroy()
        viewer_b.destroy()

    return 0


def record_experiment_parameters(exp_id, theme, rel_path, prm_freq, dim_pca, dataset, solver_array, precond_array,
                                 tol_array, max_iter, size_mat, num_data):
    # 向文件中追加内容。如果文件不存在，将会创建它。
    dir_results = rel_path + '/results'
    now = datetime.now()
    with open(dir_results + '/parameters.txt', 'a') as file:
        file.write('theme: ' + str(theme) + '\n')
        file.write('exp_id: ' + str(exp_id) + '\n')
        file.write('exp_time: ' + str(now) + '\n')
        file.write('prm_freq: ' + str(prm_freq) + '\n')
        file.write('dim_pca: ' + str(dim_pca) + '\n')
        file.write('dataset: ' + str(dataset) + '\n')
        file.write('solver_array: ' + str(solver_array) + '\n')
        file.write('precond_array: ' + str(precond_array) + '\n')
        file.write('tol_array: ' + str(tol_array) + '\n')
        file.write('max_iter: ' + str(max_iter) + '\n')
        file.write('size_mat: ' + str(size_mat) + '\n')
        file.write('num_data: ' + str(num_data) + '\n')
        file.close()
    return 0


if __name__ == "__main__":
    # 相关可调参数
    # 数据集参数FFT后截至的矩阵边长 维数是这个数的平方
    prm_freq = 12
    # PCA的维数
    dim_pca = 7

    # 本次运算使用的相关参数
    # 实验主题
    theme = '不同误差下 不同预处理下 贪心排序 possion 的 两种算法对比'
    exp_id = '20230920_1_0'
    # 所用数据集
    dataset = 'pos'
    # 数据集的大小
    num_data = 100
    # 误差设定
    # tol_array = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    tol_array = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    # tol_array = [1e-6, 1e-4]
    # tol_array = [1e-2]
    # 最大迭代次数
    max_iter = 10000
    # 矩阵大小
    s = 1
    size_mat = 7473
    # 涉及的预处理方式
    # precond_array = ["none", "icc0", "icc1", "icc2", "jacobi", "bjacobi", "sor", "eisenstat", "ilu0", "ilu1", "ilu2", "asm", "gasm", "gamg", "cholesky"]
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm", "eisenstat"]
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm"]
    precond_array = ["icc0", "icc1", "icc2", "ilu0", "ilu1", "ilu2"]
    # precond_array = ["none"]
    # 涉及的求解器
    solver_array = ["gcrodr", "gmres"]
    # solver_array = ["gmres"]
    # solver_array = ["gcrodr"]
    data_path = './recycle datasets/Poisson_generate/data/data_poisson_0.025_100_7473.mat'
    ###################################################################################################################
    # 生成文件夹 rel_path是当前文件夹的相对路径
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id)

    build_folder(exp_id, dataset, solver_array, precond_array, tol_array, max_iter, size_mat, num_data)
    # 存储实验参数
    record_experiment_parameters(exp_id, theme, rel_path, prm_freq, dim_pca, dataset, solver_array, precond_array,
                                 tol_array, max_iter,
                                 size_mat, num_data)
    # 生成对应数据集
    # A, b, w = dataset_generator(dataset, size_mat, num_data, rel_path, prm_freq, dim_pca)
    # 从matlab得到数据集
    data = scipy.io.loadmat(data_path)
    A = [item[0] if isinstance(item, list) and len(item) == 1 else item for item in data['A_data'][0]]
    b = data['b_data']
    w = [item[0] if isinstance(item, list) and len(item) == 1 else item for item in data['w_data'][0]]
    # w = torch.tensor(data['w_data'])

    # 生成坐标数组和区域划分 将矩阵放入对应的区域 生成矩阵运算顺序
    address, area_array, area_num_totel, area_seq, mat_seq = seq_generator(dataset, size_mat, num_data, rel_path,
                                                                           prm_freq, dim_pca, w)
    # address, area_array, area_num_totel, area_seq, mat_seq = seq_generator_vec(dataset, size_mat, num_data, rel_path,
    #                                                                            prm_freq, dim_pca, w)

    # 待办 将area_array, area_num_totel, area_seq, mat_seq存入指定文件夹

    # 将矩阵和向量转换为PETSc格式然后输出dat格式 按照指定顺序存入文件夹
    petsc_generator(dataset, size_mat, num_data, rel_path, A, b, mat_seq)

    print('exp date generate done')
