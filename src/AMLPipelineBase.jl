module AMLPipelineBase


include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer
export fit, fit!, transform, transform!, fit_transform, fit_transform!

include("utils.jl")
using .Utils
export holdout, kfold, score, infer_eltype
export nested_dict_to_tuples
export nested_dict_set!
export nested_dict_merge
export mergedict, getiris
export skipmean, skipmedian, skipstd
export aggregatorclskipmissing
export getiris, getprofb
export find_catnum_columns
export train_test_split


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
export |>, +, |, *, >>
export Pipeline, ComboPipeline

end # module
