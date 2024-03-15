#!/usr/bin/env python
# -*- coding:utf-8 _*-
# 此test文件测试petsc4py数据封装成dat文件
# 使用ctypes调用c函数


import matplotlib.pyplot as plt

import pickle

def plot_2_label(x_label, y_label, results):
    # 提取数据
    x_data = [res[x_label] for res in results]
    y_data = [res[y_label] for res in results]

    # 绘图
    plt.plot(x_data, y_data, marker='o', linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + 'vs' + y_label)
    plt.grid(True)
    plt.show()
    return 0

def plot_precond_tol(results):
    # 提取唯一的 solver 类型
    solvers = sorted(list(set([res['solver'] for res in results])))

    # 提取唯一的 precond 和 tol 组合
    unique_precond_tol_combinations = sorted(list(set([(res['precond'], res['tol']) for res in results])),
                                             key=lambda x: x[0])

    # 对于每个 precond 和 tol 组合，绘制一条线
    for precond, tol in unique_precond_tol_combinations:
        y_values = [res['average_time'] for res in results if res['precond'] == precond and res['tol'] == tol]
        plt.plot(solvers, y_values, marker='o', label=f'precond={precond}, tol={tol}')

    plt.xlabel('Solver')
    plt.ylabel('Average Time')
    plt.title('Average Time vs. Solver')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 0

def plot_precond_tol_up(results):
    # 提取唯一的 solver 类型
    solvers = sorted(list(set([res['solver'] for res in results])))

    # 提取唯一的 precond 和 tol 组合
    unique_precond_tol_combinations = sorted(list(set([(res['precond'], res['tol']) for res in results])),
                                             key=lambda x: x[0])

    for precond, tol in unique_precond_tol_combinations:
        y_values = [res['average_time'] for res in results if res['precond'] == precond and res['tol'] == tol]

        # 只添加上升或下降的线
        if is_increasing(y_values):
            plt.plot(solvers, y_values, marker='o', label=f'precond={precond}, tol={tol}')

    plt.xlabel('Solver')
    plt.ylabel('Average Time')
    plt.title('Average Time vs. Solver')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 0

def plot_precond_tol_down(results):
    # 提取唯一的 solver 类型
    solvers = sorted(list(set([res['solver'] for res in results])))

    # 提取唯一的 precond 和 tol 组合
    unique_precond_tol_combinations = sorted(list(set([(res['precond'], res['tol']) for res in results])),
                                             key=lambda x: x[0])

    for precond, tol in unique_precond_tol_combinations:
        y_values = [res['average_time'] for res in results if res['precond'] == precond and res['tol'] == tol]

        # 只添加上升或下降的线
        if is_decreasing(y_values):
            plt.plot(solvers, y_values, marker='o', label=f'precond={precond}, tol={tol}')

    plt.xlabel('Solver')
    plt.ylabel('Average Time')
    plt.title('Average Time vs. Solver')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 0

def is_increasing(y_values):
    return all(y1 <= y2 for y1, y2 in zip(y_values, y_values[1:]))

def is_decreasing(y_values):
    return all(y1 >= y2 for y1, y2 in zip(y_values, y_values[1:]))



if __name__ == "__main__":
    # 相关可调参数
    # 数据集参数FFT后截至的矩阵边长 维数是这个数的平方
    prm_freq = 12
    # PCA的维数
    dim_pca = 7

    # 本次运算使用的相关参数
    # 实验主题
    theme = '不同误差下 不同预处理下 darcy_rectangular_pwc 的 两种算法对比'
    exp_id = '20230910'
    # 所用数据集
    dataset = 'possion2d_mpi'
    # 数据集的大小
    num_data = 1000
    # 误差设定
    tol_array = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # tol_array = [1e-6, 1e-4]
    # tol_array = [1e-2]
    # 最大迭代次数
    max_iter = 10000
    # 矩阵大小
    s = 102
    size_mat = (s - 2) * (s - 2)
    # 涉及的预处理方式
    # precond_array = ["none", "icc0", "icc1", "icc2", "jacobi", "bjacobi", "sor", "eisenstat", "ilu0", "ilu1", "ilu2", "asm", "gasm", "gamg", "cholesky"]
    precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm", "gamg"]
    # precond_array = ["icc0", "icc1", "icc2", "ilu0", "ilu1", "ilu2", "eisenstat", "cholesky"]
    # precond_array = ["none"]
    # 涉及的求解器
    solver_array = ["gcrodr","gmres"]
    # solver_array = ["gmres"]
    # solver_array = ["gcrodr"]

    ###################################################################################################################
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id)

    # 读取数据集
    dir_results = rel_path + '/results'
    with open(dir_results + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
        f.close()


    # x_label = 'solver'
    # y_label = 'average_time'
    # plot_2_label(x_label, y_label, results)

    # plot_precond_tol(results)
    plot_precond_tol_down(results)







    print('exp plot done')

    # result_exp = {
    #     'solver': solver,
    #     'precond': precond,
    #     'tol': tol,
    #     'max_iter': max_iter,
    #     'size_mat': size_mat,
    #     'num_data': num_data,
    #     'total_time': total_time,
    #     'average_time': average_time,
    #     'total_iter': total_iter,
    #     'average_iter': average_iter,
    #     'max_iter_count': max_iter_count,
    #     'total_rnorm': total_rnorm
    # }
    # results.append(result_exp)
