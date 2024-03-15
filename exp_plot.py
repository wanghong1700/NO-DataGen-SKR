#!/usr/bin/env python
# -*- coding:utf-8 _*-
# 此test文件测试petsc4py数据封装成dat文件
# 使用ctypes调用c函数

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat

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


def plot_precond_tol_iter(results):
    import matplotlib.pyplot as plt

    # 假设 results 已经被填充了数据
    # results = [...]

    # 提取唯一的 solver 类型
    solvers = sorted(list(set([res['solver'] for res in results])))

    # 提取唯一的 precond 和 tol 组合
    unique_precond_tol_combinations = sorted(list(set([(res['precond'], res['tol']) for res in results])),
                                             key=lambda x: x[0])

    # 对于每个 precond 和 tol 组合，绘制一条线
    for precond, tol in unique_precond_tol_combinations:
        y_values = [res['average_iter'] for res in results if res['precond'] == precond and res['tol'] == tol]
        plt.plot(solvers, y_values, marker='o', label=f'precond={precond}, tol={tol}')

    plt.xlabel('Solver')
    plt.ylabel('Average Iteration')
    plt.title('Average Iteration vs. Solver')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 0

def is_increasing(y_values):
    return all(y1 <= y2 for y1, y2 in zip(y_values, y_values[1:]))

def is_decreasing(y_values):
    return all(y1 >= y2 for y1, y2 in zip(y_values, y_values[1:]))


def plot_results(results, x_axis_type, y_axis_type, trend='all'):
    """
    根据给定的 x 轴和 y 轴类型绘制 results 中的数据。

    参数:
    - results: 数据列表
    - x_axis_type: x 轴的类型 (例如: 'solver')
    - y_axis_type: y 轴的类型 (例如: 'average_iter')
    - trend: 'up' 代表只绘制上升的线, 'down' 代表只绘制下降的线, 'all' 代表绘制所有线
    """
    # 提取唯一的 x 轴数据
    x_unique_values = sorted(list(set([res[x_axis_type] for res in results])))

    # 提取唯一的 precond 和 tol 组合
    unique_precond_tol_combinations = sorted(list(set([(res['precond'], res['tol']) for res in results])),
                                             key=lambda x: x[0])

    # 对于每个 precond 和 tol 组合，绘制一条线
    for precond, tol in unique_precond_tol_combinations:
        y_values = [res[y_axis_type] for res in results if res['precond'] == precond and res['tol'] == tol]

        # 根据 trend 参数进行筛选
        if trend == 'up' and not is_increasing(y_values):
            continue
        if trend == 'down' and not is_decreasing(y_values):
            continue

        plt.plot(x_unique_values, y_values, marker='o', label=f'precond={precond}, tol={tol}')

    plt.xlabel(x_axis_type)
    plt.ylabel(y_axis_type)
    plt.title(f'{y_axis_type} vs. {x_axis_type}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return 0


def plot_3d_ratio(results, x_axis_type, y_axis_type, z_axis_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取所有唯一的 x 和 y 轴数据，并为其创建编码
    x_unique_values = list(set([res[x_axis_type] for res in results]))
    y_unique_values = list(set([res[y_axis_type] for res in results]))

    x_encoded = {val: idx for idx, val in enumerate(x_unique_values)}
    y_encoded = {val: idx for idx, val in enumerate(y_unique_values)}

    X, Y, Z = [], [], []

    for x in x_unique_values:
        for y in y_unique_values:
            gmres_val = next((res[z_axis_type] for res in results if
                              res[x_axis_type] == x and res[y_axis_type] == y and res['solver'] == 'gmres'), None)
            gcrodr_val = next((res[z_axis_type] for res in results if
                               res[x_axis_type] == x and res[y_axis_type] == y and res['solver'] == 'gcrodr'), None)

            # 计算 gmres 和 gcrodr 的比值
            if gmres_val is not None and gcrodr_val is not None and gcrodr_val != 0:
                ratio = gmres_val / gcrodr_val
                X.append(x_encoded[x])
                Y.append(y_encoded[y])
                Z.append(ratio)

    ax.scatter(X, Y, Z, c='r', marker='o')
    ax.set_xticks(list(x_encoded.values()))
    ax.set_xticklabels(list(x_encoded.keys()))

    ax.set_yticks(list(y_encoded.values()))
    ax.set_yticklabels(list(y_encoded.keys()))

    ax.set_xlabel(x_axis_type)
    ax.set_ylabel(y_axis_type)
    ax.set_zlabel(z_axis_type + ' ratio (gmres/gcrodr)')
    plt.show()
    return X, Y, Z


def save_as_mat(X, Y, Z, filename='data.mat'):
    """
    将三维数据保存为.mat文件。
    """
    data_dict = {
        'X': X,
        'Y': Y,
        'Z': Z
    }
    savemat(filename, data_dict)



def plot_tol_vs_max_iter(results):
    # 获取所有不同的solver和precond
    solvers = list(set([res['solver'] for res in results]))
    preconds = list(set([res['precond'] for res in results]))

    # 定义一个形状列表，为每个solver分配不同的形状
    # 如果有更多的solver类型，可以扩展这个列表
    markers = ['x', '+', '^', 'v', '<', '>', 'p', '*']

    # 定义线型列表，为每个solver分配不同的线型
    linestyles = ['-', '--', '-.', ':']

    # 确保我们有足够的形状和线型用于所有的solver
    if len(solvers) > len(markers) or len(solvers) > len(linestyles):
        raise ValueError("需要为每个solver定义更多的形状和线型!")

    plt.figure(figsize=(10, 6))

    # 为每个solver绘图
    for idx, solver in enumerate(solvers):
        for precond in preconds:
            # 为当前solver和precond提取tol和max_iter_count数据
            tol_values = [res['tol'] for res in results if res['solver'] == solver and res['precond'] == precond]
            max_iter_count_values = [res['max_iter_count'] for res in results if
                                     res['solver'] == solver and res['precond'] == precond]

            # 如果这种组合有多个数据点，我们就画出来
            if len(tol_values) > 1:
                # 使用semilogx绘制散点图，并为当前solver指定形状和线型
                plt.semilogx(tol_values, max_iter_count_values, linestyle=linestyles[idx], marker=markers[idx],
                             label=f"{solver} with {precond}")

    plt.xlabel('Tol')
    plt.ylabel('Max Iter Count')
    plt.title('Max Iter Count vs Tol for Different Solvers and Preconds')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.show()


def plot_tol_vs_max_iter_for_gcrodr(results):
    # 获取所有不同的precond
    preconds = list(set([res['precond'] for res in results]))

    # 定义一个形状列表，为gcrodr分配形状
    marker = 'x'

    # 定义线型为gcrodr
    linestyle = '--'

    plt.figure(figsize=(10, 6))

    for precond in preconds:
        # 为gcrodr和precond提取tol和max_iter_count数据
        tol_values = [res['tol'] for res in results if res['solver'] == 'gcrodr' and res['precond'] == precond]
        max_iter_count_values = [res['max_iter_count'] for res in results if
                                 res['solver'] == 'gcrodr' and res['precond'] == precond]

        # 如果这种组合有多个数据点，我们就画出来
        if len(tol_values) > 1:
            # 使用semilogx绘制散点图，并为gcrodr指定形状和线型
            plt.semilogx(tol_values, max_iter_count_values, linestyle=linestyle, marker=marker,
                         label=f"gcrodr with {precond}")

    plt.xlabel('Tol')
    plt.ylabel('Max Iter Count')
    plt.title('Max Iter Count vs Tol for gcrodr with Different Preconds')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.show()

def find_matching_results(results1, results2):
    matches = []

    for r1 in results1:
        for r2 in results2:
            if (r1['precond'] == r2['precond'] and
                r1['tol'] == r2['tol'] and
                r1['solver'] == r2['solver']):
                matches.append((r1, r2))

    return matches


def compare_and_plot(matches):
    average_time_ratios = []
    average_iter_ratios = []
    labels = []

    for r1, r2 in matches:
        average_time_ratio = r1['average_time'] / r2['average_time']
        average_iter_ratio = r1['average_iter'] / r2['average_iter']

        average_time_ratios.append(average_time_ratio)
        average_iter_ratios.append(average_iter_ratio)

        label = f"{r1['precond']} {r1['tol']} {r1['solver']}"
        labels.append(label)

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(average_time_ratios, average_iter_ratios)

    # Annotate each point with its label
    for i, label in enumerate(labels):
        plt.annotate(label, (average_time_ratios[i], average_iter_ratios[i]), fontsize=8)

    plt.xlabel('average_time ratio (File1/File2)')
    plt.ylabel('average_iter ratio (File1/File2)')
    plt.title('Comparison of average_time and average_iter ratios')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_and_plot_gcrodr(matches):
    average_time_ratios = []
    average_iter_ratios = []
    labels = []

    for r1, r2 in matches:
        if r1['solver'] == 'gcrodr':
            continue
        if r1['precond'] == 'gamg':
            continue

        average_time_ratio = r1['average_time'] / r2['average_time']
        average_iter_ratio = r1['average_iter'] / r2['average_iter']

        average_time_ratios.append(average_time_ratio)
        average_iter_ratios.append(average_iter_ratio)

        label = f"{r1['precond']} {r1['tol']} {r1['solver']}"
        # label = ''
        labels.append(label)

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(average_time_ratios, average_iter_ratios)

    # Annotate each point with its label
    for i, label in enumerate(labels):
        plt.annotate(label, (average_time_ratios[i], average_iter_ratios[i]), fontsize=8)

    plt.xlabel('average_time ratio (File1/File2)')
    plt.ylabel('average_iter ratio (File1/File2)')
    plt.title('Comparison of average_time and average_iter ratios')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    # exp_id = '20230915_1'
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id)
    # num_data = 100
    # exp_id_0 = '20230915'
    # rel_path_0 = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id_0)

    # 读取数据集
    dir_results = rel_path + '/results'
    with open(dir_results + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
        f.close()

    # # 读取数据集
    # dir_results_0 = rel_path_0 + '/results'
    # with open(dir_results_0 + '/results.pkl', 'rb') as f:
    #     results_0 = pickle.load(f)
    #     f.close()

    # matches = find_matching_results(results, results_0)
    # compare_and_plot(matches)
    # compare_and_plot_gcrodr(matches)


    # x_label = 'solver'
    # y_label = 'average_time'
    # plot_2_label('tol', 'max_iter_count', results)
    # plot_tol_vs_max_iter(results)
    # plot_tol_vs_max_iter_for_gcrodr(results)

    # plot_results(results, 'solver', 'average_iter', 'down')

    # 调用plot_3d_ratio函数并接收返回的X, Y, Z
    # X, Y, Z = plot_3d_ratio(results, 'precond', 'tol', 'average_iter')


    # 使用这些数组调用save_as_mat函数
    # ave_as_mat(X, Y, Z, 'max_iter_count_data.mat')

    # X, Y, Z = plot_3d_ratio(results, 'precond', 'tol', 'average_time')

    # 使用这些数组调用save_as_mat函数
    # save_as_mat(X, Y, Z, 'average_time_ratio_data.mat')



    # plot_precond_tol(results)
    # plot_precond_tol_down(results)

    # plot_precond_tol_iter(results)


    for precond in precond_array:
        for solver in solver_array:

            tols = []
            gmres_average_times = []
            gmres_average_iters = []
            gcrodr_average_times = []
            gcrodr_average_iters = []

            for result in results:
                if result['solver'] == 'gmres' and result['precond'] == precond:
                    tols.append(result['tol'])
                    gmres_average_times.append(result['average_time'])
                    gmres_average_iters.append(result['average_iter'])

                if result['solver'] == 'gcrodr' and result['precond'] == precond:
                    tols.append(result['tol'])
                    gcrodr_average_times.append(result['average_time'])
                    gcrodr_average_iters.append(result['average_iter'])

        # 使用\t分隔并打印结果
        print(precond)
        print("\t".join(map(str, tols)))
        print("\t".join(map(str, gmres_average_times)))
        print("\t".join(map(str, gcrodr_average_times)))
        print("\t".join(map(str, gmres_average_iters)))
        print("\t".join(map(str, gcrodr_average_iters)))





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
