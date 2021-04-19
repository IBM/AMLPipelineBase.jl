module AMLPipelineBase

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer
export fit, fit!, transform, transform!, fit_transform, fit_transform!

include("utils.jl")
using .Utils
using AMLPipelineBase.Utils
export holdout, kfold, score, infer_eltype, 
       nested_dict_to_tuples, 
       nested_dict_set!, 
       nested_dict_merge, 
       create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       getiris, getprofb,
       train_test_split


include("baselinemodels.jl")
using .BaselineModels
export Baseline, Identity

include("basefilters.jl")
using .BaseFilters
export Imputer, OneHotEncoder, Wrapper

include("featureselector.jl")
using .FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

include("decisiontree.jl")
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost

include("ensemble.jl")
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate, pipe_performance

include("naremover.jl")
using .NARemovers
export NARemover

include("pipelines.jl")
using .Pipelines
export @pipeline, @pipelinex, @pipelinez
export |>, +, |, *
export Pipeline, ComboPipeline

end # module
