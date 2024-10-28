# Decision trees as found in DecisionTree Julia package.
module DecisionTreeLearners

import DecisionTree
DT = DecisionTree
# standard included modules
using DataFrames: DataFrame, nrow
using Random

using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!

export PrunedTree, RandomForest, Adaboost

# Pruned CART decision tree.

"""
    PrunedTree(
      Dict(
        :purity_threshold => 1.0,
        :max_depth => -1,
        :min_samples_leaf => 1,
        :min_samples_split => 2,
        :min_purity_increase => 0.0
      )
    )

Decision tree classifier.  
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:purity_threshold` => 1.0 (merge leaves having >=thresh combined purity)
- `:max_depth` => -1 (maximum depth of the decision tree)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
mutable struct PrunedTree <: Learner
    name::String
    model::Dict{Symbol,Any}

    function PrunedTree(args=Dict())
        default_args = Dict{Symbol,Any}(
            :name => "prunetree",
            # Output to train against
            # (:class).
            :output => :class,
            # Options specific to this implementation.
            :impl_args => Dict{Symbol,Any}(
                # Merge leaves having >= purity_threshold CombineMLd purity.
                :purity_threshold => 1.0,
                # Maximum depth of the decision tree (default: no maximum).
                :max_depth => -1,
                # Minimum number of samples each leaf needs to have.
                :min_samples_leaf => 1,
                # Minimum number of samples in needed for a split.
                :min_samples_split => 2,
                # Minimum purity needed for a split.
                :min_purity_increase => 0.0
            )
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        new(cargs[:name], cargs)
    end
end

function PrunedTree(name::String; opt...)
    PrunedTree(Dict(:name => name, :impl_args => Dict(pairs(opt))))
end

"""
    fit!(tree::PrunedTree, features::DataFrame, labels::Vector) 

Optimize the hyperparameters of `PrunedTree` instance.
"""
function fit!(ptree::PrunedTree, features::DataFrame, labels::Vector)::Nothing
    @assert nrow(features) == length(labels)
    instances = Matrix(features)
    args = ptree.model[:impl_args]
    btreemodel = DT.build_tree(
        labels,
        instances,
        0, # num_subfeatures (keep all)
        args[:max_depth],
        args[:min_samples_leaf],
        args[:min_samples_split],
        args[:min_purity_increase])
    btreemodel = DT.prune_tree(btreemodel, args[:purity_threshold])
    ptree.model[:dtmodel] = btreemodel
    ptree.model[:impl_args] = args
    return nothing
end

function fit(ptree::PrunedTree, features::DataFrame, labels::Vector)::PrunedTree
    fit!(ptree, features, labels)
    return deepcopy(ptree)
end

"""
    transform!(ptree::PrunedTree, features::DataFrame)

Predict using the optimized hyperparameters of the trained `PrunedTree` instance.
"""
function transform!(ptree::PrunedTree, features::DataFrame)::Vector
    isempty(features) && return []
    instances = Matrix(features)
    model = ptree.model[:dtmodel]
    return DT.apply_tree(model, instances)
end

function transform(ptree::PrunedTree, features::DataFrame)::Vector
    return transform!(ptree, features)
end

# Random forest (CART).

"""
    RandomForest(
      Dict(
        :output => :class,
        :num_subfeatures => 0,
        :num_trees => 10,
        :partial_sampling => 0.7,
        :max_depth => -1
      )
    )

Random forest classification. 
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:num_subfeatures` => 0  (number of features to consider at random per split)
- `:num_trees` => 10 (number of trees to train)
- `:partial_sampling` => 0.7 (fraction of samples to train each tree on)
- `:max_depth` => -1 (maximum depth of the decision trees)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
mutable struct RandomForest <: Learner
    name::String
    model::Dict{Symbol,Any}

    function RandomForest(args=Dict())
        default_args = Dict{Symbol,Any}(
            :name => "rf",
            # Output to train against
            # (:class).
            :output => :class,
            # Options specific to this implementation.
            :impl_args => Dict{Symbol,Any}(
                # Number of features to train on with trees (default: 0, keep all).
                :num_subfeatures => 0,
                # Number of trees in forest.
                :num_trees => 10,
                # Proportion of trainingset to be used for trees.
                :partial_sampling => 0.7,
                # Maximum depth of each decision tree (default: no maximum).
                :max_depth => -1
            )
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        new(cargs[:name], cargs)
    end
end

function RandomForest(name::String; opt...)
    RandomForest(Dict(:name => name, :impl_args => Dict(pairs(opt))))
end

"""
    fit!(forest::RandomForest, features::DataFrame, labels::Vector) 

Optimize the parameters of the `RandomForest` instance.
"""
function fit!(forest::RandomForest, features::DataFrame, labels::Vector)::Nothing
    @assert nrow(features) == length(labels)
    instances = Matrix(features)
    # Set training-dependent options
    impl_args = forest.model[:impl_args]
    # Build model
    model = DT.build_forest(
        labels,
        instances,
        impl_args[:num_subfeatures],
        impl_args[:num_trees],
        impl_args[:partial_sampling],
        impl_args[:max_depth]
    )
    forest.model[:dtmodel] = model
    forest.model[:impl_args] = impl_args
    return nothing
end

function fit(forest::RandomForest, features::DataFrame, labels::Vector)::RandomForest
    fit!(forest, features, labels)
    return deepcopy(forest)
end



"""
    transform!(forest::RandomForest, features::DataFrame) 


Predict using the optimized hyperparameters of the trained `RandomForest` instance.
"""
function transform!(forest::RandomForest, features::DataFrame)::Vector
    isempty(features) && return []
    instances = features
    instances = Matrix(features)
    model = forest.model[:dtmodel]
    return DT.apply_forest(model, instances)
end

function transform(forest::RandomForest, features::DataFrame)::Vector
    return transform!(forest, features)
end

# Adaboosted decision stumps.

"""
    Adaboost(
      Dict(
        :output => :class,
        :num_iterations => 7
      )
    )

Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:
- `:num_iterations` => 7 (number of iterations of AdaBoost)

Implements `fit!`, `transform!`
"""
mutable struct Adaboost <: Learner
    name::String
    model::Dict{Symbol,Any}

    function Adaboost(args=Dict())
        default_args = Dict(
            :name => "adaboost",
            # Output to train against
            # (:class).
            :output => :class,
            # Options specific to this implementation.
            :impl_args => Dict(
                # Number of boosting iterations.
                :num_iterations => 7
            )
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        new(cargs[:name], cargs)
    end
end

function Adaboost(name::String; opt...)
    Adaboost(Dict(:name => name, :impl_args => Dict(pairs(opt))))
end

"""
    fit!(adaboost::Adaboost, features::DataFrame, labels::Vector) 

Optimize the hyperparameters of `Adaboost` instance.
"""
function fit!(adaboost::Adaboost, features::DataFrame, labels::Vector)::Nothing
    @assert nrow(features) == length(labels)
    instances = Matrix(features)
    # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
    #              This differs to DecisionTree
    #              official documentation to avoid confusion in variable
    #              naming within CombineML.
    ensemble, coefficients = DT.build_adaboost_stumps(
        labels, instances, adaboost.model[:impl_args][:num_iterations]
    )
    adaboost.model[:ensemble] = ensemble
    adaboost.model[:coefficients] = coefficients
    return nothing
end

function fit(adaboost::Adaboost, features::DataFrame, labels::Vector)::Adaboost
    fit!(adaboost, features, labels)
    return deepcopy(adaboost)
end

"""
    transform!(adaboost::Adaboost, features::DataFrame)

Predict using the optimized hyperparameters of the trained `Adaboost` instance.
"""
function transform!(adaboost::Adaboost, features::DataFrame)::Vector
    isempty(features) && return []
    instances = Matrix(features)
    return DT.apply_adaboost_stumps(
        adaboost.model[:ensemble], adaboost.model[:coefficients], instances
    )
end

function transform(adaboost::Adaboost, features::DataFrame)::Vector
    return transform!(adaboost, features)
end

end # module
