#!/usr/bin/env python
# -*- coding:utf-8 _*-
# 此test文件测试petsc4py数据封装成dat文件
# 使用ctypes调用c函数

import numpy as np

import os

from datetime import datetime
import time


def petsc_solver(dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter):
    # 获取当前权限
    # st = os.stat('./e')

    # 添加执行权限
    # os.chmod('./e', st.st_mode | stat.S_IEXEC)
    dir = rel_path + '/data_{}_{}_{}_PETSc'.format(dataset, size_mat, num_data)
    dir_results = rel_path + '/results'

    for solver in solver_array:
        for precond in precond_array:
            precond_cmd = precond
            cmd_run = "mpirun --allow-run-as-root -n 20 ./e"
            # cmd_run = "./e"
            if precond == 'cholesky':
                cmd_run = "./e"
            elif precond == 'ilu0':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 0"
            elif precond == 'ilu1':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 1"
            elif precond == 'ilu2':
                cmd_run = "./e"
                precond_cmd = "ilu -pc_factor_levels 2"
            elif precond == 'eisenstat':
                precond_cmd = "sor -pc_sor_variant eisenstat"
            elif precond == 'icc0':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 0"
            elif precond == 'icc1':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 1"
            elif precond == 'icc2':
                cmd_run = "./e"
                precond_cmd = "icc -pc_factor_levels 2"

            for tol in tol_array:
                start_time = time.perf_counter()

                if solver == 'gmres':
                    dir_x = rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat, num_data)
                    dir_output = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol,
                                                                                         max_iter,
                                                                                         size_mat, num_data)
                    cmd = cmd_run + ' -ksp_converged_reason -pc_type {} -ksp_rtol {} -ksp_gmres_restart 40 -ksp_type hpddm ' \
                          '-nmat {} -load_dir {} -load_dir_x {} -load_dir_output ' \
                          '{} -ksp_max_it {}' \
                        .format(precond_cmd, tol, num_data, dir, dir_x, dir_output, max_iter)


                if solver == 'gcrodr' :
                    dir_x = rel_path + '/data_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)
                    dir_output = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)
                    cmd = cmd_run + ' -ksp_converged_reason -pc_type {} -ksp_rtol {} -ksp_gmres_restart 40 -ksp_type hpddm ' \
                          '-ksp_hpddm_type {} -ksp_hpddm_recycle 20 -nmat {} -load_dir {} -load_dir_x {} -load_dir_output ' \
                          '{} -ksp_max_it {}'\
                        .format(precond_cmd, tol, solver, num_data, dir, dir_x, dir_output, max_iter)

                print(cmd)
                record_experiment_start(dir_results, cmd, dataset, solver, precond, tol, max_iter, size_mat,
                                       num_data)
                # subprocess.run(cmd, shell=True)
                os.system(cmd)
                record_experiment_end(dir_results, dataset, solver, precond, tol, max_iter, size_mat, num_data)

                end_time = time.perf_counter()
                total_time = end_time - start_time
                average_time = total_time / num_data
                rel_path_total = rel_path + '/output/output_{}_{}_{}_{}_{}_{}_{}/total'.format(dataset, solver, precond, tol, max_iter,
                                                                           size_mat,num_data)

                with open(rel_path_total + '/total_time.txt', 'w') as file:
                    file.write(str(total_time))
                    file.close()
                with open(rel_path_total + '/average_time.txt', 'w') as file:
                    file.write(str(average_time))
                    file.close()

                total_iter = []
                with open(rel_path_total + '/output_total_iter.txt', 'r') as file:
                    for line in file:
                        iter = int(line.strip())  # 使用float()或int()根据数字类型转换
                        total_iter.append(iter)
                    file.close()
                max_iter_count = total_iter.count(max_iter)
                with open(rel_path_total + '/max_iter_count.txt', 'w') as file:
                    file.write(str(max_iter_count))
                    file.close()
                average_iter = np.mean(total_iter)
                with open(rel_path_total + '/average_iter.txt', 'w') as file:
                    file.write(str(average_iter))
                    file.close()

                print(dataset, solver, precond, tol, max_iter, size_mat, num_data, 'done')
    return 0

def record_experiment_parameters(theme, rel_path, prm_freq, dim_pca, dataset, solver_array, precond_array, tol_array, max_iter, size_mat, num_data):
    # 向文件中追加内容。如果文件不存在，将会创建它。
    dir_results = rel_path + '/results'
    now = datetime.now()
    with open(dir_results + '/parameters.txt', 'a') as file:
        file.write('theme: ' + str(theme) + '\n')
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

def record_experiment_start(dir_results, cmd, dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data):
    now = datetime.now()
    with open(dir_results + '/exp_record.txt', 'a') as file:
        output = "exp_start: {}, {}, {}, {}, {}, {}, {}\n".format(dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data)
        file.write(output)
        file.write('exp_start_cmd: ' + str(cmd) + '\n')
        file.write('exp_start_time: ' + str(now) + '\n')
        file.close()
    return 0

def record_experiment_end(dir_results, dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data):
    now = datetime.now()
    with open(dir_results + '/exp_record.txt', 'a') as file:
        file.write('exp_end_time: ' + str(now) + '\n')
        output = "exp_end: {}, {}, {}, {}, {}, {}, {}\n".format(dataset, solver, precond, tol, max_iter, size_mat,
                                                               num_data)
        file.write(output)
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
    exp_id = '20230920_0_0'
    # 所用数据集
    dataset = 'pos'
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
    s = 1
    size_mat = 497
    # 涉及的预处理方式
    # precond_array = ["none", "icc0", "icc1", "icc2", "jacobi", "bjacobi", "sor", "eisenstat", "ilu0", "ilu1", "ilu2", "asm", "gasm", "gamg", "cholesky"]
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm", "gamg", "eisenstat"]
    # precond_array = ["none", "jacobi", "bjacobi", "sor", "asm", "gasm"]
    precond_array = ["icc0", "icc1", "icc2", "ilu0", "ilu1", "ilu2"]
    # precond_array = ["none"]
    # 涉及的求解器
    solver_array = ["gcrodr", "gmres"]
    # solver_array = ["gmres"]
    # solver_array = ["gcrodr"]
    data_path = './recycle datasets/Poisson_generate/data/data_poisson_0.1_100_497.mat'
    ###################################################################################################################
    rel_path = './data/data_{}_{}_{}'.format(dataset, num_data, exp_id)

    petsc_solver(dataset, size_mat, num_data, rel_path, solver_array, precond_array, tol_array, max_iter)

    print('exp solve done')

