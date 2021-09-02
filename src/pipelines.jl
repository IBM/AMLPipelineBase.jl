module Pipelines

using DataFrames
using Random

using ..AbsTypes
using ..BaseFilters
using ..Utils
using ..EnsembleMethods: BestLearner

import Base: |>, +, |, *, >>
export |>, +, |, *

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export Pipeline, ComboPipeline 
export @pipeline, @pipelinex, @pipelinez
export processexpr!,sympipeline

"""
    Pipeline(machs::Vector{<:Machine},args::Dict=Dict())

Linear pipeline which iteratively calls and passes the result
of `fit_transform` to the succeeding elements in the pipeline.

Implements `fit!` and `transform!`.
"""
mutable struct Pipeline <: Workflow
   name::String
   model::Dict{Symbol,Any}

   function Pipeline(args::Dict = Dict())
      default_args = Dict{Symbol,Any}(
         :name => "linearpipeline",
         # machines as list to chain in sequence.
         :machines => Vector{Machine}(),
         # Transformer args as list applied to same index transformer.
         :machine_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    Pipeline(machs::Vector{<:Machine},args::Dict=Dict())

Helper function for Pipeline structure.
"""
function Pipeline(machs::Vector{<:Machine},args::Dict)
   Pipeline(Dict(:machines => machs, args...))
end

function Pipeline(machs::Vector{<:Machine};opt...)
   Pipeline(Dict(:machines=>machs,:machine_args=>Dict(pairs(opt))))
end

"""
    Pipeline(machs::Vararg{Machine})

Helper function for Pipeline structure.
"""
function Pipeline(machs::Vararg{<:Machine})
   (eltype(machs) <: Machine) || throw(ArgumentError("argument setup error"))
   v=[x for x in machs] # convert tuples to vector
   combo = Pipeline(v)
   return combo
end

function fit!(pipe::Pipeline, features::DataFrame=DataFrame(), labels::Vector=[])::Nothing
   instances=deepcopy(features)
   machines = pipe.model[:machines]
   machine_args = pipe.model[:machine_args]

   current_instances = instances
   new_machines = Machine[]

   # fit-transform all except last element
   # last element calls fit only
   trlength = length(machines)
   for t_index in 1:(trlength - 1)
      machine = createmachine(machines[t_index], machine_args)
      push!(new_machines, machine)
      fit!(machine, current_instances, labels)
      current_instances = transform!(machine, current_instances)
   end
   machine = createmachine(machines[trlength], machine_args)
   push!(new_machines, machine)
   fit!(machine, current_instances, labels)

   pipe.model[:machines] = new_machines
   pipe.model[:machine_args] = machine_args
   return nothing
end

function fit(pipe::Pipeline, features::DataFrame=DataFrame(), labels::Vector=[])::Pipeline
   fit!(pipe, features, labels)
   return deepcopy(pipe)
end

function transform!(pipe::Pipeline, instances::DataFrame=DataFrame())::Union{Vector, DataFrame}
   machines = pipe.model[:machines]

   current_instances = deepcopy(instances)
   for t_index in 1:length(machines)
      machine = machines[t_index]
      current_instances = transform!(machine, current_instances)
   end

   return current_instances
end

function transform(pipe::Pipeline, instances::DataFrame=DataFrame())::Union{Vector, DataFrame}
   return transform!(pipe, instances)
end

"""
    ComboPipeline(machs::Vector{T}) where {T<:Machine}

Feature union pipeline which iteratively calls 
`fit_transform` of each element and concatenate
their output into one dataframe.

Implements `fit!` and `transform!`.
"""
mutable struct ComboPipeline <: Workflow
   name::String
   model::Dict{Symbol,Any}

   function ComboPipeline(args::Dict = Dict())
      default_args = Dict{Symbol,Any}(
         :name => "combopipeline",
         # machines as list to chain in sequence.
         :machines => Vector{Machine}(),
         # Transformer args as list applied to same index transformer.
         :machine_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

function ComboPipeline(machs::Vector{<:Machine},args::Dict) 
   ComboPipeline(Dict(:machines => machs, args...))
end

function ComboPipeline(machs::Vector{<:Machine};opt...)
   ComboPipeline(Dict(:machines=>machs,:machine_args=>Dict(pairs(opt))))
end

function ComboPipeline(machs::Vararg{<:Machine})
   (eltype(machs) <: Machine) || throw(ArgumentError("argument setup error"))
   v=[eval(x) for x in machs] # convert tuples to vector
   combo = ComboPipeline(v)
   return combo
end

function fit!(pipe::ComboPipeline, features::DataFrame, labels::Vector=[])::Nothing
  instances=deepcopy(features)
  machines = pipe.model[:machines]
  machine_args = pipe.model[:machine_args]

  new_machines = Machine[]
  new_instances = DataFrame()
  trlength = length(machines)
  for t_index in eachindex(machines)
    machine = createmachine(machines[t_index], machine_args)
    push!(new_machines, machine)
    fit!(machine, instances, labels)
  end

  pipe.model[:machines] = new_machines
  pipe.model[:machine_args] = machine_args
  return nothing
end

function fit(pipe::ComboPipeline, features::DataFrame, labels::Vector=[])::ComboPipeline
   fit!(pipe, features, labels)
   return deepcopy(pipe)
end

function transform!(pipe::ComboPipeline, features::DataFrame=DataFrame())::Union{Vector,DataFrame}
  isempty(features) && return []
  machines = pipe.model[:machines]
  instances = deepcopy(features)
  new_instances = DataFrame()
  for t_index in eachindex(machines)
    machine = machines[t_index]
    current_instances = transform!(machine, instances) 
    new_instances = mcat(new_instances,current_instances)
  end
  return new_instances
end

# dispatch concat between vectors/dataframes
function mcat(x::DataFrame,y::DataFrame)
   hcat(x,y,makeunique=true)
end

function mcat(x::DataFrame,y::Vector)
   hcat(x,DataFrame(v=y),makeunique=true)
end

function mcat(x::Vector,y::DataFrame)
   hcat(DataFrame(v=x),y,makeunique=true)
end

function transform(pipe::ComboPipeline, features::DataFrame=DataFrame())::Union{Vector,DataFrame}
   return transform!(pipe, features)
end

function processexpr!(args::AbstractVector)
  for ndx in eachindex(args)
    if typeof(args[ndx]) == Expr
      processexpr!(args[ndx].args)
    elseif args[ndx] == :(|>)
      args[ndx] = :Pipeline
    elseif args[ndx] == :+
      args[ndx] = :ComboPipeline
    elseif args[ndx] == :*
      args[ndx] = :BestLearner
    end
  end
  return args
end

# check if quoted expression 
macro pipeline(expr)
  lexpr = copy(:($(esc(expr))))
  if expr isa Expr && expr.head === :quote
    lexpr = :($(esc(expr.args[1])))
  end
  args=processexpr!(lexpr.args)
  lexpr.args=args
  lexpr
end

# unwrap symbol
macro pipelinez(sexpr)
  @assert sexpr isa Symbol
  quote
	 lexpr = copy($(esc(sexpr)))
	 processexpr!(lexpr.args)	
	 lexpr
  end 
end

macro pipelinex(expr)
  lexpr = copy(:($(esc(expr))))
  if expr isa Expr && expr.head === :quote
    lexpr = :($(esc(expr.args[1])))
  end
  res=processexpr!(lexpr.args)
  :($res[1])
end

function sympipeline(pexpr)
  expr = copy(pexpr)
  processexpr!(expr.args)
  expr
end

|>(a::Machine, b::Machine) = Pipeline([a,b])
>>(a::Machine, b::Machine) = Pipeline([a,b])
+(a::Machine, b::Machine)  = ComboPipeline([a,b])
*(a::Machine, b::Machine)  = BestLearner([a,b])
|(a::Machine, b::Machine)  = BestLearner([a,b])

end
