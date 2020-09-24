module AMLPBase

greet() = print("Hello World!")

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer

include("utils.jl")
using .Utils
export getiris, getprofb

include("decisiontree.jl")
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost

include("ensemble.jl")
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

include("pipelines.jl")
using .Pipelines
export @pipeline, @pipelinex, @pipelinez
export Pipeline, ComboPipeline

end # module
