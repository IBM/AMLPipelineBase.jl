module EnsembleMethods

using Statistics
import StatsBase
import IterTools: product
import MLBase

# standard included modules
using DataFrames: DataFrame, nrow
using Random

using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export VoteEnsemble, StackEnsemble, BestLearner

using ..DecisionTreeLearners


"""
    VoteEnsemble(
       Dict( 
          # Output to train against
          # (:class).
          :output => :class,
          # Learners in voting committee.
          :learners => [PrunedTree(), Adaboost(), RandomForest()]
       )
    )

Set of machine learners employing majority vote to decide prediction.

Implements: `fit!`, `transform!`
"""
mutable struct VoteEnsemble <: Learner
  name::String
  model::Dict{Symbol,Any}
  
  function VoteEnsemble(args=Dict())
     default_args = Dict{Symbol,Any}( 
      :name => "votingens",
      # Output to train against
      # (:class).
      :output => :class,
      # Learners in voting committee.
      :learners => [PrunedTree(), Adaboost(), RandomForest()]
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],cargs)
  end
end

function VoteEnsemble(learners::Vector{<:Learner}, opt::Dict)
   VoteEnsemble(Dict(:learners => learners, opt...))
end

function VoteEnsemble(learners::Vector{<:Learner}; opt...)
   VoteEnsemble(Dict(:learners => learners, Dict(pairs(opt))...))
end

function VoteEnsemble(learners::Vararg{Learner})
   (eltype(learners) <: Learner) || throw(ArgumentError("argument setup error"))
   v=[x for x in learners] # convert tuples to vector
   VoteEnsemble(v)
end

"""
    fit!(ve::VoteEnsemble, instances::DataFrame, labels::Vector)

Training phase of the ensemble.
"""
function fit!(ve::VoteEnsemble, instances::DataFrame, labels::Vector)::Nothing
  @assert nrow(instances) == length(labels)
  # Train all learners
  learners = ve.model[:learners]
  for learner in learners
    fit!(learner, instances, labels)
  end
  ve.model[:learners] = learners 
  return nothing
end

function fit(ve::VoteEnsemble, instances::DataFrame, labels::Vector)::VoteEnsemble
   fit!(ve, instances, labels)
   return deepcopy(ve)
end

"""
    transform!(ve::VoteEnsemble, instances::DataFrame)

Prediction phase of the ensemble.
"""
function transform!(ve::VoteEnsemble, instances::DataFrame)::Vector
  isempty(instances) && return []
  # Make learners vote
  learners = ve.model[:learners]
  predictions = map(learner -> transform!(learner, instances), learners)
  # Return majority vote prediction
  return StatsBase.mode(predictions)
end

function transform(ve::VoteEnsemble, instances::DataFrame)::Vector
   return transform!(ve,instances)
end

"""
    StackEnsemble(
       Dict(    
          # Output to train against
          # (:class).
          :output => :class,
          # Set of learners that produce feature space for stacker.
          :learners => [PrunedTree(), Adaboost(), RandomForest()],
          # Machine learner that trains on set of learners' outputs.
          :stacker => RandomForest(),
          # Proportion of training set left to train stacker itself.
          :stacker_training_proportion => 0.3,
          # Provide original features on top of learner outputs to stacker.
          :keep_original_features => false
       )
    )

An ensemble where a 'stack' of learners is used for training and prediction.
"""
mutable struct StackEnsemble <: Learner
  name::String
  model::Dict{Symbol,Any}

  function StackEnsemble(args=Dict())
     default_args = Dict{Symbol,Any}(
      :name => "stackens",
      # Output to train against
      # (:class).
      :output => :class,
      # Set of learners that produce feature space for stacker.
      :learners => [PrunedTree(), Adaboost(), RandomForest()],
      # Machine learner that trains on set of learners' outputs.
      :stacker => RandomForest(),
      # Proportion of training set left to train stacker itself.
      :stacker_training_proportion => 0.3,
      # Provide original features on top of learner outputs to stacker.
      :keep_original_features => false
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],cargs)
  end
end

function StackEnsemble(learners::Vector{<:Learner}, opt::Dict)
   StackEnsemble(Dict(:learners => learners, opt...))
end

function StackEnsemble(learners::Vector{<:Learner}; opt...)
   StackEnsemble(Dict(:learners => learners, Dict(pairs(opt))...))
end

function StackEnsemble(learners::Vararg{Learner})
   (eltype(learners) <: Learner) || throw(ArgumentError("argument setup error"))
   v=[x for x in learners] # convert tuples to vector
   StackEnsemble(v)
end

"""
    fit!(se::StackEnsemble, instances::DataFrame, labels::Vector)

Training phase of the stack of learners.

- perform holdout to obtain indices for 
- partition learner and stacker training sets
- partition training set for learners and stacker
- train all learners
- train stacker on learners' outputs
- build final model from the trained learners
"""
function fit!(se::StackEnsemble, instances::DataFrame, labels::Vector)::Nothing
  @assert nrow(instances) == length(labels)
  learners = se.model[:learners]
  num_learners = size(learners, 1)
  num_instances = size(instances, 1)
  num_labels = size(labels, 1)
  
  # Perform holdout to obtain indices for 
  # partitioning learner and stacker training sets
  shuffled_indices = randperm(num_instances)
  stack_proportion = se.model[:stacker_training_proportion]
  (learner_indices, stack_indices) = holdout(num_instances, stack_proportion)
  
  # Partition training set for learners and stacker
  learner_instances = instances[learner_indices, :]
  stack_instances = instances[stack_indices, :]
  learner_labels = labels[learner_indices]
  stack_labels = labels[stack_indices]
  
  # Train all learners
  for learner in learners
    fit!(learner, learner_instances, learner_labels)
  end
  
  # Train stacker on learners' outputs
  label_map = MLBase.labelmap(labels)
  stacker = se.model[:stacker]
  keep_original_features = se.model[:keep_original_features]
  stacker_instances = build_stacker_instances(learners, stack_instances, 
                                              label_map, keep_original_features) |> x->DataFrame(x,:auto)
  fit!(stacker, stacker_instances, stack_labels)
  
  # Build model
  se.model[:learners] = learners
  se.model[:stacker] = stacker 
  se.model[:label_map] = label_map 
  se.model[:keep_original_features] = keep_original_features
  return nothing
end

function fit(se::StackEnsemble, instances::DataFrame, labels::Vector)::StackEnsemble
   fit!(se,instances,labels)
   return deepcopy(se)
end

"""
    transform!(se::StackEnsemble, instances::DataFrame)

Build stacker instances and predict
"""
function transform!(se::StackEnsemble, instances::DataFrame)::Vector
  isempty(instances) && return []
  # Build stacker instances
  learners = se.model[:learners]
  stacker = se.model[:stacker]
  label_map = se.model[:label_map]
  keep_original_features = se.model[:keep_original_features]
  stacker_instances = build_stacker_instances(learners, instances, 
                                              label_map, keep_original_features) |> x-> DataFrame(x,:auto)

  # Predict
  return transform!(stacker, stacker_instances)
end

function transform(se::StackEnsemble, instances::DataFrame)::Vector
   transform!(se,instances)
end

# Build stacker instances.
function build_stacker_instances(
  learners::Vector{T}, instances::DataFrame, 
  label_map, keep_original_features=false) where T<:Learner

  # Build empty stacker instance space
  num_labels = size(label_map.vs, 1)
  num_instances = size(instances, 1)
  num_learners = size(learners, 1)
  stacker_instances = zeros(num_instances, num_learners * num_labels)

  # Fill stack instances with predictions from learners
  for l_index = 1:num_learners
    predictions = transform!(learners[l_index], instances)
    for p_index in 1:size(predictions, 1)
      pred_encoding = MLBase.labelencode(label_map, predictions[p_index])
      pred_column = (l_index-1) * num_labels + pred_encoding
      stacker_instances[p_index, pred_column] = 
        one(eltype(stacker_instances))
    end
  end

  # Add original features to stacker instance space if enabled
  if keep_original_features
    stacker_instances = [instances stacker_instances]
  end
  
  # Return stacker instances
  return stacker_instances
end

"""
    BestLearner(
       Dict(
          # Output to train against
          # (:class).
          :output => :class,
          # Function to return partitions of instance indices.
          :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
          # Function that selects the best learner by index.
          # Arg learner_partition_scores is a (learner, partition) score matrix.
          :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, dims=2))[2],      
          # Score type returned by score() using respective output.
          :score_type => Real,
          # Candidate learners.
          :learners => [PrunedTree(), Adaboost(), RandomForest()],
          # Options grid for learners, to search through by BestLearner.
          # Format is [learner_1_options, learner_2_options, ...]
          # where learner_options is same as a learner's options but
          # with a list of values instead of scalar.
          :learner_options_grid => nothing
       )
    )

Selects best learner from the set by performing a 
grid search on learners if grid option is indicated.
"""
mutable struct BestLearner <: Learner
  name::String
  model::Dict{Symbol,Any}
  
  function BestLearner(args=Dict{Symbol,Any}())
     default_args = Dict{Symbol,Any}(
      :name => "bestens",
      # Output to train against
      # (:class).
      :output => :class,
      # Function to return partitions of instance indices.
      :partition_generator => (instances, labels) -> kfold(size(instances, 1), 5),
      # Function that selects the best learner by index.
      # Arg learner_partition_scores is a (learner, partition) score matrix.
      :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, dims=2))[2],      
      # Score type returned by score() using respective output.
      :score_type => Real,
      # Candidate learners.
      :learners => [PrunedTree(), Adaboost(), RandomForest()],
      # Options grid for learners, to search through by BestLearner.
      # Format is [learner_1_options, learner_2_options, ...]
      # where learner_options is same as a learner's options but
      # with a list of values instead of scalar.
      :learner_options_grid => nothing
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],cargs)
  end
end


function BestLearner(learners::Vector{<:Learner}, opt::Dict)
   BestLearner(Dict(:learners => learners, opt...))
end

function BestLearner(learners::Vector{<:Learner}; opt...)
   BestLearner(Dict(:learners => learners, Dict(pairs(opt))...))
end

function BestLearner(learners::Vararg{Learner})
   (eltype(learners) <: Learner) || throw(ArgumentError("argument setup error"))
   v=[x for x in learners] # convert tuples to vector
   BestLearner(v)
end


"""
    fit!(bls::BestLearner, instances::DataFrame, labels::Vector)

Training phase:

- obtain learners as is if grid option is not present 
- generate learners if grid option is present 
- foreach prototype learner, generate learners with specific options found in grid
- generate partitions
- train each learner on each partition and obtain validation output
"""
function fit!(bls::BestLearner, instances::DataFrame, labels::Vector)::Nothing
  @assert nrow(instances) == length(labels)
  # Obtain learners as is if no options grid present 
  if bls.model[:learner_options_grid] == nothing
    learners = bls.model[:learners]
  # Generate learners if options grid present 
  else
    # Foreach prototype learner, generate learners with specific options
    # found in grid.
    learners = Transformer[]
    for l_index in 1:length(bls.model[:learners])
      # Obtain options grid
      options_prototype = bls.model[:learner_options_grid][l_index]
      grid_list = nested_dict_to_tuples(options_prototype)
      grid_keys = map(x -> x[1], grid_list)
      grid_values = map(x -> x[2], grid_list)

      # Foreach combination of options
      # generate learner.
      for combination in product(grid_values...)
        # Assign values for each option
        learner_options = deepcopy(options_prototype)
        for g_index in 1:length(grid_list)
          keys = grid_keys[g_index]
          value = combination[g_index]
          nested_dict_set!(learner_options, keys, value)
        end

        # Generate learner
        learner_prototype = bls.model[:learners][l_index]
        learner = create_transformer(learner_prototype, learner_options)

        # Append to candidate learners
        push!(learners, learner)
      end
    end
  end

  # Generate partitions
  partition_generator = bls.model[:partition_generator]
  partitions = partition_generator(instances, labels)

  # Train each learner on each partition and obtain validation output
  num_partitions           = size(partitions, 1)
  num_learners             = size(learners, 1)
  num_instances            = size(instances, 1)
  score_type               = bls.model[:score_type]
  learner_partition_scores = Array{score_type}(undef,num_learners, num_partitions)
  for l_index = 1:num_learners, p_index = 1:num_partitions
    partition = partitions[p_index]
    rest = setdiff(1:num_instances, partition)
    learner = learners[l_index]

    training_instances   = instances[partition,:]
    training_labels      = labels[partition]
    validation_instances = instances[rest, :]
    validation_labels    = labels[rest]

    fit!(learner, training_instances, training_labels)
    predictions = transform!(learner, validation_instances)
    result = score(:accuracy, validation_labels, predictions)
    learner_partition_scores[l_index, p_index] = result
  end
  
  # Find best learner based on selection function
  best_learner_index = 
    bls.model[:selection_function](learner_partition_scores)
  best_learner = learners[best_learner_index]
  
  # Retrain best learner on all training instances
  fit!(best_learner, instances, labels)
  
  # Create model
  bls.model[:best_learner]             = best_learner
  bls.model[:best_learner_index]       = best_learner_index
  bls.model[:learners]                 = learners
  bls.model[:learner_partition_scores] = learner_partition_scores
  return nothing
end

function fit(bls::BestLearner, instances::DataFrame, labels::Vector)::BestLearner
   fit!(bls,instances,labels)
   return deepcopy(bls)
end

""" 
    transform!(bls::BestLearner, instances::DataFrame)

Choose the best learner based on cross-validation results and use it for prediction.
"""
function transform!(bls::BestLearner, instances::DataFrame)::Vector
   isempty(instances) && return []
   transform!(bls.model[:best_learner], instances)
end

function transform(bls::BestLearner, instances::DataFrame)::Vector
   transform!(bls,instances)
end

end # module
