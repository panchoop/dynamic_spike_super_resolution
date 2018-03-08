### Phase-space super-resolution
# By Francisco Romero and Timothee Wintz.
v0.1

Codes to simulate examples to which apply the super resolution reconstruction 
theory on moving particles. The paper of reference is not yet published, but has
as authors Giovanni Alberti, Habib Ammari, Francisco Romero, Timothee Wintz.

The codes require the SparseInverProblems solver from https://github.com/nboyd/SparseInverseProblems.jl, we copied it and included small modifications that let it abort to avoid entering infinite loops.

There are simulations in one and two dimensions, the parameters to generate them can be modified in the files 1d_parameters.jl and 2d_parameters.jl respectively. The script to generate them are Main_1dSim.jl and Main_2dSim.jl respectively.
Be advised: generation of examples is lengthy and it is recommended to use run these scripts in a parallel fashion.

The generated data is stored in "/data", inside them organized by type of simulation and date of it.

The scripts to generate the figures with the results are plots_1d_results.py and plots_2d_results.py respectively. Inside these scripts, at the beggining, it is possible to change the example to simulate (in /data).

