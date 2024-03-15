#!/usr/bin/env python
# -*- coding:utf-8 _*-
# 此test文件测试petsc4py数据封装成dat文件
# 使用ctypes调用c函数

import numpy as np

import pickle
from datetime import datetime
import time

def result_generate(results, dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter):
    # 获取当前权限
    # st = os.stat('./e')

    # 添加执行权限
    # os.chmod('./e', st.st_mode | stat.S_IEXEC)
    # dir = rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data)
    # dir_results = rel_path + '/results'

    for solver in solver_array:
        for precond in precond_array:
            for tol in tol_array:

                rel_path_total = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)

                with open(rel_path_total + '/total_time.txt', 'r') as file:
                    total_time = float(file.read())
                    file.close()
                with open(rel_path_total + '/average_time.txt', 'r') as file:
                    average_time = float(file.read())
                    file.close()

                total_iter = []
                with open(rel_path_total + '/output_total_iter.txt', 'r') as file:
                    for line in file:
                        iter = int(line.strip())  # 使用float()或int()根据数字类型转换
                        total_iter.append(iter)
                    file.close()

                with open(rel_path_total + '/max_iter_count.txt', 'r') as file:
                    max_iter_count = int(file.read())
                    file.close()

                with open(rel_path_total + '/average_iter.txt', 'r') as file:
                    average_iter = float(file.read())
                    file.close()

                total_rnorm = []
                with open(rel_path_total + '/output_total_rnorm.txt', 'r') as file:
                    for line in file:
                        rnorm = float(line.strip())  # 使用float()或int()根据数字类型转换
                        total_rnorm.append(rnorm)
                    file.close()

                result_exp= {
                    'solver': solver,
                    'precond': precond,
                    'tol': tol,
                    'max_iter': max_iter,
                    'size_mat': size_mat,
                    'num_data': num_data,
                    'total_time': total_time,
                    'average_time': average_time,
                    'total_iter': total_iter,
                    'average_iter': average_iter,
                    'max_iter_count':max_iter_count,
                    'total_rnorm': total_rnorm
                }
                results.append(result_exp)
                print(dataset, solver, precond, tol, max_iter, size_mat, num_data, 'done')

    dir_results = rel_path + '/results'
    with open(dir_results + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)
        f.close()

    return results

if __name__ == "__main__":
    # 相关可调参数
    # 数据集参数FFT后截至的矩阵边长 维数是这个数的平方
    prm_freq = 12
    # PCA的维数
    dim_pca = 7

    # 本次运算使用的相关参数
    # 实验主题
    theme = '不同误差下 不同预处理下 贪心排序 darcy_rectangular_pwc 的 两种算法对比'
    exp_id = '20230915_2'
    # 所用数据集
    dataset = 'possion2d_mpi'
    # 数据集的大小
    num_data = 100
    # 误差设定
    tol_array = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # tol_array = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # tol_array = [1e-6, 1e-4]
    # tol_array = [1e-2]
    # 最大迭代次数
    max_iter = 10000
    # 矩阵大小
    s = 102
    size_mat = (s - 2) * (s - 2)
    # 涉及的预处理方式
    # precond_array = ["none", "icc0", "icc1", "icc2", "jacobi", "bjacobi", "sor", "eisenstat", "ilu0", "ilu1", "ilu2", "asm", "gasm", "gamg", "cholesky"]
    precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm", "gamg", "eisenstat"]
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm"]
    # precond_array = ["icc0", "icc1", "icc2", "ilu0", "ilu1", "ilu2", "cholesky"]
    # precond_array = ["none"]
    # 涉及的求解器
    solver_array = ["gcrodr", "gmres"]
    ###################################################################################################################
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id)

    results = []

    results = result_generate(results, dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter)

    print('exp results generate done')

