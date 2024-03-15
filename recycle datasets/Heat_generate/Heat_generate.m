clear;
close all
import pde.*;

num = 1;
n = 3; % n 是阶数
p_h = 0.05;

A_data = {};
b_data = [];
w_data = {};
r_data = {};

for i=1:num

    thermalmodelS = createpde("thermal","steadystate");
    
    r1 = [3 4 -.5 .5 .5 -.5  -.8 -.8 .8 .8];
    r2 = [3 4 -.05 .05 .05 -.05  -.4 -.4 .4 .4];
    gdm = [r1; r2]';
    
    g = decsg(gdm,'R1-R2',['R1'; 'R2']');
    
    geometryFromEdges(thermalmodelS,g);
    
    figure
    pdegplot(thermalmodelS,"EdgeLabels","on"); 
    axis([-.9 .9 -.9 .9]);
    title("Block Geometry With Edge Labels Displayed")
    r1 = 100 *rand();
    r2 = -100 *rand();

    thermalBC(thermalmodelS,"Edge",6,"Temperature",r1);
    thermalBC(thermalmodelS,"Edge",1,"HeatFlux",r2);
  
    [f_bc_1, w_1] = chebfun_generate(n);
    f_handle_1 = @(x) feval(f_bc_1, x);
    r3 = 100 *rand();
    thermalProperties(thermalmodelS,"ThermalConductivity",1);
    % thermalProperties(thermalmodelS,"ThermalConductivity",1);
    
    generateMesh(thermalmodelS,"Hmax",p_h);

    % solvepde.m 76 -> solveStationary,m 56
    femodel = assembleFEMatrices(thermalmodelS);
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
    KK = thermalmodelS.checkForSymmetry(KK);

    A_data{end+1} = KK;
    b_data(end+1, :) = FF;
    w_data{end+1} = w_1;
    r0 = [];
    r0(end+1, :) = r1;
    r0(end+1, :) = r2;
    r_data{end+1} = r0;

    % load('data_generate.mat');
    % A_data{end+1} = A;
    % b_data(end+1, :) = b;
    % w0 = [];
    % w0(end+1, :) = w_1;
    % w0(end+1, :) = w_2;
    % w_data{end+1} = w0;
    % delete('data_generate.mat');

    figure 
    pdeplot(thermalmodelS); 
    axis equal
    title("Block With Finite Element Mesh Displayed")
    % 
    R = solve(thermalmodelS);
    T = R.Temperature;
    figure
    pdeplot(thermalmodelS,"XYData",T,"Contour","on","ColorMap","hot"); 
    axis equal
    title("Temperature, Steady State Solution")

end

filename = ['data_heat_' num2str(p_h) '_1500.mat'];
save(filename, 'A_data', 'b_data', 'w_data', 'r_data');
