% clear;
% close all
import pde.*;

num = 1;
n = 5; % n 是阶数
p_h = 0.005;

A_data = {};
b_data = [];
w_data = {};

for i=1:num

    % 定义矩形域
    rect = [3; 4; 0; 1; 1; 0; 0; 0; 1; 1];
    g = decsg(rect);
    model = createpde();
    
    geometryFromEdges(model, g);
    
    % 设置PDE系数
    c = 1;
    a = 0;
    f = -1;
    [f_bc_0, w_0] = chebfun_generate(n);
    f_handle_0 = @(x) feval(f_bc_0, x);
    specifyCoefficients(model, 'm', 0, 'd', 0, 'c', c, 'a', a, 'f', @(loc,state) f_handle_0(loc.x));


    [f_bc_1_0, w_1] = chebfun_generate(n);
    f_handle_1_0 = @(x) feval(f_bc_1_0, x);
    [f_bc_2_0, w_2] = chebfun_generate(n);
    f_handle_2_0 = @(x) feval(f_bc_2_0, x);
    [f_bc_3_0, w_3] = chebfun_generate(n);
    f_handle_3_0 = @(x) feval(f_bc_3_0, x);
    [f_bc_4_0, w_4] = chebfun_generate(n);
    f_handle_4_0 = @(x) feval(f_bc_4_0, x);

    f_handle_1 = @(x) f_handle_1(x) + 1/10*f_handle_1_0(x);
    f_handle_2 = @(x) f_handle_2(x) + 1/10*f_handle_2_0(x);
    f_handle_3 = @(x) f_handle_3(x) + 1/10*f_handle_3_0(x);
    f_handle_4 = @(x) f_handle_4(x) + 1/10*f_handle_4_0(x);
    
    % 应用边界条件
    applyBoundaryCondition(model, 'dirichlet', 'edge', 1, 'u', @(loc,state) f_handle_1(loc.x));
    applyBoundaryCondition(model, 'dirichlet', 'edge', 2, 'u', @(loc,state) f_handle_2(loc.x));
    applyBoundaryCondition(model, 'dirichlet', 'edge', 3, 'u', @(loc,state) f_handle_3(loc.x));
    applyBoundaryCondition(model, 'dirichlet', 'edge', 4, 'u', @(loc,state) f_handle_4(loc.x));
    
    % 网格和求解
    generateMesh(model, 'Hmax', p_h);

    % solvepde.m 76 -> solveStationary,m 56
    femodel = assembleFEMatrices(model);
    K = femodel.K;
    A = femodel.A;
    F = femodel.F;
    Q = femodel.Q;
    G = femodel.G;
    H = femodel.H;
    R = femodel.R;
    [null,orth]=pdenullorth(H);
    if size(orth,2)==0
        ud=zeros(size(K,2),1);
    else
        ud=full(orth*((H*orth)\R));
    end

    KK=K+A+Q;
    FF=null'*((F+G)-KK*ud);
    KK=null'*KK*null;
    KK = model.checkForSymmetry(KK);

    A_data{end+1} = KK;
    b_data(end+1, :) = FF;
    w0 = [];
    w0(end+1, :) = w_0;
    w0(end+1, :) = w_1;
    w0(end+1, :) = w_2;
    w0(end+1, :) = w_3;
    w0(end+1, :) = w_4;
    w_data{end+1} = w0;

    result = solvepde(model);

    % 绘图
    figure;
    pdeplot(model, 'XYData', result.NodalSolution, 'Contour', 'on');
    title('Poisson Equation Solution')
    xlabel('x')
    ylabel('y')
   saveas(gcf, 'Possion3_5.eps', 'epsc'); 
end

filename = ['data_poisson_' num2str(p_h) '_100.mat'];
save(filename, 'A_data', 'b_data', 'w_data');



