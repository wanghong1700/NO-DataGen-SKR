ENVIRONMENT:
1. Proper installation of the PETSc and MPI libraries (C version) is required.
2. Necessary Python libraries mentioned in the files, especially the Python versions of PETSc and MPI, must be correctly installed.

NOTE:
1. Please modify the parallel count in 'exp_solve' to ensure it's less than or equal to the number of logical cores on your computer.

CODE STRUCTURE:
1. Run 'make e.c' to generate the 'e' executable file.
2. Run 'exp_generate' to produce linear equation problems.
3. Run 'exp_solve' to solve the linear equation problems.
4. Run 'exp_result' to collect computational statistics based on the solutions.
5. Run 'exp_plot' to analyze the data results.

This directory provides executable codes in sequence. For detailed experimental scripts, refer to the respective folders.

In this context:
recycle datasets generates the required linear equation systems using MATLAB.
code_ablation contains the ablation study code.
code_basic is the foundational code template.
code_dracy pertains to the Darcy flow dataset code.
code_heat addresses the Thermal Problem dataset code.
code_hmh is for the Helmholtz Equation dataset code.
code_pos relates to the Poisson equation dataset code.
