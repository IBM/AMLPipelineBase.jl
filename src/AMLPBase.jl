module AMLPBase

greet() = print("Hello World!")

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer
export fit!, transform!, fit_transform!

include("utils.jl")
using .Utils
export getiris, getprofb

include("baselinemodels.jl")
using .BaselineModels
export Baseline, Identity

include("basefilters.jl")
using .BaseFilters
export OneHotEncoder, Imputer

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

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate

end # module
