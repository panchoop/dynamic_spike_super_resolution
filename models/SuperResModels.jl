module SuperResModels
using SparseInverseProblemsMod
using SparseInverseProblemsMod.Util
using NLopt
import SparseInverseProblemsMod: psi, dpsi, getStartingPoint, parameterBounds, phi, computeGradient
export SuperRes, DynamicSuperRes, Conv1d, DynamicConv1d, Fourier1d, DynamicFourier1d, Fourier2d, DynamicFourier2d, is_in_bounds, dim,setBounds,psi, Conv2d, Conv2dParameters, DynamicConv2d, DynamicConv2dParameters

include("SuperRes.jl")
include("DynamicSuperRes.jl")
include("Conv1d.jl")
include("Conv2d.jl")
include("Fourier1d.jl")
include("Fourier2d.jl")
end
