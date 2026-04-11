module ACEfriction

# utility functions for conversion of arrays, manipulation of bases and generation of bases for bond environments
include("./utils/utils.jl")
# utility functions for importing and internally storing data of friction tensors/matrices 
include("./datautils.jl")

include("./atomcutoffs.jl")
include("./matrixmodels/matrixmodels.jl")
include("./frictionmodels.jl")
include("./frictionfit/frictionfit.jl")
include("./matrixmodelsutils.jl")

import ACEfriction.FrictionModels: FrictionModel, Gamma, Sigma
export Gamma, Sigma, FrictionModel

import ACEfriction.FrictionFit: FrictionData, FluxFrictionModel, flux_assemble
export FrictionData, FluxFrictionModel, flux_assemble

import ACEfriction.MatrixModels: RWCMatrixModel, mbdpd_matrixmodel, OnsiteOnlyMatrixModel, PWCMatrixModel
export RWCMatrixModel, mbdpd_matrixmodel, OnsiteOnlyMatrixModel, PWCMatrixModel

import ACEfriction.DataUtils: load_h5fdata, save_h5fdata
import ACEfrictionCore: write_dict, read_dict
export write_dict, read_dict, load_h5fdata, save_h5fdata
import NeighbourLists
import YAML, JSON

#import JuLIP: Atoms
#export Atoms

import ACEfrictionCore.ACEbonds: EllipsoidCutoff
export EllipsoidCutoff, SphericalCutoff

import ACEfrictionCore: Invariant, EuclideanVector, EuclideanMatrix
export Invariant, EuclideanVector, EuclideanMatrix

function load_dict(fname::AbstractString)
   if endswith(fname, ".json")
      return JSON.load_json(fname)
   elseif endswith(fname, ".yaml") || endswith(fname, ".yml")
      return YAML.load_yaml(fname)
   else
      @warn("Unrecognised file format. Expected: \"*.json\" or \"*.yaml\", got filename: $(fname); default to json format")
      return JSON.load_json(fname)# throw(error("Unrecognised file format. Expected: \"*.json\" or \"*.yaml\", got filename: $(fname)"))
      # throw(error("Unrecognised file format. Expected: \"*.json\" or \"*.yaml\", got filename: $(fname)"))
   end
end

function save_dict(fname::AbstractString, D::AbstractDict; indent=0)
   if endswith(fname, ".json")
      return JSON.save_json(fname, D; indent=indent)
   elseif endswith(fname, ".yaml") || endswith(fname, ".yml")
      return YAML.save_yaml(fname, D)
   else
      @warn("Unrecognised file format. Expected: \"*.json\" or \"*.yaml\", got filename: $(fname); default to json format")
      # throw(error("Unrecognised file format. Expected: \"*.json\" or \"*.yaml\", got filename: $(fname)"))
      return JSON.save_json(fname, D; indent=indent)
   end
end

export save_dict, load_dict

import ACEfriction.FrictionFit: weighted_l2_loss, weighted_l1_loss
export weighted_l2_loss, weighted_l1_loss
end
