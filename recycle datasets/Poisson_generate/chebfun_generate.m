function [f, coefficients] = chebfun_generate(n)
    % n 是阶数
    % 生成随机系数，范围可以根据需要修改
    coefficients = 2*rand(n+1, 1) - 1; % 生成-1到1之间的随机数
    
    % 生成随机的5阶切比雪夫多项式函数
    f = chebfun(0);
    for k = 0:n
        T_k = chebpoly(k);  % 获取k阶切比雪夫多项式
        f = f + coefficients(k+1) * T_k;
    end
end
