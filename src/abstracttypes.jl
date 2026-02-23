module AbsTypes

using DataFrames

export fit, fit!, transform, transform!, fit_transform, fit_transform!
export Machine, Learner, Transformer, Workflow, Computer

abstract type Machine end
abstract type Computer <: Machine end # does computation: learner and transformer
abstract type Workflow <: Machine end # types: Linear vs Combine
abstract type Learner <: Computer end
abstract type Transformer <: Computer end

"""
fit!(mc::Machine, input::DataFrame, output::Vector)

Generic trait to be overloaded by different subtypes of Machine.
Multiple dispatch for fit!.
"""
function fit!(mc::Machine, ::DataFrame, ::Vector)::Nothing
  throw(ArgumentError(typeof(mc), " not implemented"))
end

function fit(mc::Machine, ::DataFrame, ::Vector)::Machine
  throw(ArgumentError(typeof(mc), " not implemented"))
end


"""
transform!(mc::Machine, input::DataFrame)

Generic trait to be overloaded by different subtypes of Machine.
Multiple dispatch for transform!.
"""
function transform!(mc::Machine, ::DataFrame)::Union{DataFrame,Vector}
  throw(ArgumentError(typeof(mc), " not implemented"))
end

function transform(mc::Machine, ::DataFrame)::Union{DataFrame,Vector}
  throw(ArgumentError(typeof(mc), " not implemented"))
end


"""
fit_transform!(mc::Machine, input::DataFrame, output::Vector)

Dynamic dispatch that calls in sequence `fit!` and `transform!` functions.
"""
function fit_transform!(mc::Machine, input::DataFrame=DataFrame(), output::Vector=Vector())::Union{Vector,DataFrame}
  fit!(mc, input, output)
  transform!(mc, input)
end

function fit_transform(mc::Machine, input::DataFrame=DataFrame(), output::Vector=Vector())::Union{Vector,DataFrame}
  rmc = fit(mc, input, output)
  transform(rmc, input)
end

end

