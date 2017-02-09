module SuperResModels
using SparseInverseProblems
using SparseInverseProblems.Util
using NLopt
import SparseInverseProblems: psi, dpsi, getStartingPoint, parameterBounds
export SuperRes, DynamicSuperRes, Conv1d, DynamicConv1d, Fourier1d, DynamicFourier1d, Fourier2d, DynamicFourier2d, is_in_bounds, dim,setBounds,psi
include("SuperRes.jl")
include("DynamicSuperRes.jl")
include("Conv1d.jl")
include("Fourier1d.jl")
include("Fourier2d.jl")
end
