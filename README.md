### Phase-space super-resolution
v0.1

Codes to simulate examples to which apply the super resolution reconstruction 
theory on moving particles. The paper of reference is not yet published, but has
as authors Giovanni Alberti, Habib Ammari, Francisco Romero, Timothee Wintz.

WARNING: The code requires the module https://github.com/nboyd/SparseInverseProblems.jl to be installed. Due to what it seems to be a compatibility issue, it works with Julia 0.6.0 but with Julia 0.6.2 (current version) it seems to enter endless loops for certain examples, possibly because of some solver modification.

There are simulations in one and two dimensions, the parameters to generate them can be modified in the files 1d_parameters.jl and 2d_parameters.jl respectively. The script to generate them are Main_1dSim.jl and Main_2dSim.jl respectively.
Be advised: generation of examples is lengthy and it is recommended to use run these scripts in a parallel fashion.

The scripts to generate the figures with the results are plots_1d_results.py and plots_2d_results.py respectively. (as for now, due to a recent rework, the script for the 1d results is incomplete).

All generated data is stored in "/data", inside them organized by type of simulation and date of it. 

There has been some hacking on the Sparse solver module, the log on this changes is in SparseModHackingLog.txt

