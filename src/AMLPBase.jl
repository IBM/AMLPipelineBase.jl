module AMLPBase

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer
export fit!, transform!, fit_transform!

include("utils.jl")
using .Utils
using AMLPBase.Utils
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       getiris, getprofb

include("baselinemodels.jl")
using .BaselineModels
export Baseline, Identity

include("basefilters.jl")
using .BaseFilters
export Imputer, OneHotEncoder, Wrapper

include("featureselector.jl")
using .FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

include("normalizer.jl")
using .Normalizers
export Normalizer

include("decisiontree.jl")
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost

include("ensemble.jl")
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

#include("svm.jl")
#using .SVMModels
#export SVMModel, svmlearners

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate

include("naremover.jl")
using .NARemovers
export NARemover

include("pipelines.jl")
using .Pipelines
export @pipeline, @pipelinex, @pipelinez
export Pipeline, ComboPipeline

end # module
