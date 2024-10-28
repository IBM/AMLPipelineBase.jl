module BaselineModels

using Random
using DataFrames: DataFrame, nrow
using StatsBase: mode

using ..Utils
using ..AbsTypes

import ..AbsTypes: fit, fit!, transform, transform!

export fit, fit!, transform, transform!
export Baseline, Identity

"""
    Baseline(
       default_args = Dict(
	       :name => "baseline",
          :output => :class,
          :strat => mode
       )
    )

Baseline model that returns the mode during classification.
"""
mutable struct Baseline <: Learner
    name::String
    model::Dict{Symbol,Any}

    function Baseline(args=Dict())
        default_args = Dict{Symbol,Any}(
            :name => "baseline",
            :output => :class,
            :strat => mode
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        new(cargs[:name], cargs)
    end
end

"""
    Baseline(name::String,opt...)

 Helper function
"""
function Baseline(name::String; opt...)
    Baseline(Dict(:name => name, Dict(pairs(opt))...))
end


"""
    fit!(bsl::Baseline,x::DataFrame,y::Vector)

Get the mode of the training data.
"""
function fit!(bsl::Baseline, x::DataFrame, y::Vector)::Nothing
    @assert nrow(x) == length(y)
    bsl.model[:choice] = bsl.model[:strat](y)
    return nothing
end

function fit(bsl::Baseline, x::DataFrame, y::Vector)::Baseline
    fit!(bsl, x, y)
    return deepcopy(bsl)
end

"""
    transform!(bsl::Baseline,x::DataFrame)

Return the mode in classification.
"""
function transform!(bsl::Baseline, x::DataFrame)::Vector
    isempty(x) && return []
    fill(bsl.model[:choice], size(x, 1))
end

function transform(bsl::Baseline, x::DataFrame)::Vector
    return transform!(bsl, x)
end

"""
    Identity(args=Dict())

Returns the input as output.
"""
mutable struct Identity <: Transformer
    name::String
    model::Dict{Symbol,Any}

    function Identity(args=Dict())
        default_args = Dict{Symbol,Any}(
            :name => "identity"
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        new(cargs[:name], cargs)
    end
end

"""
    Identity(name::String,opt...)

 Helper function
"""
function Identity(name::String)
    Baseline(Dict(:name => name))
end

"""
    fit!(idy::Identity,x::DataFrame,y::Vector)

Does nothing.
"""
function fit!(idy::Identity, x::DataFrame=DataFrame(), y::Vector=[])::Nothing
    nothing
end

function fit(idy::Identity, x::DataFrame=DataFrame(), y::Vector=[])::Identity
    return idy
end

"""
    transform!(idy::Identity,x::DataFrame)

Return the input as output.
"""
function transform!(idy::Identity, x::DataFrame=DataFrame())::DataFrame
    return x
end

function transform(idy::Identity, x::DataFrame=DataFrame())::DataFrame
    return x
end

end
