module AMLPipelineBase

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        include("abstracttypes.jl")
    end
end
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer
export fit, fit!, transform, transform!, fit_transform, fit_transform!

@setup_workload begin
    @compile_workload begin
        include("utils.jl")
    end
end
using .Utils
export holdout, kfold, score, infer_eltype, 
       nested_dict_to_tuples, 
       nested_dict_set!, 
       nested_dict_merge, 
       create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       getiris, getprofb,
       find_catnum_columns,
       train_test_split


@setup_workload begin
    @compile_workload begin
        include("baselinemodels.jl")
    end
end
using .BaselineModels
export Baseline, Identity

include("basefilters.jl")
using .BaseFilters
export Imputer, OneHotEncoder, Wrapper

@setup_workload begin
    @compile_workload begin
        include("featureselector.jl")
    end
end
using .FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

@setup_workload begin
    @compile_workload begin
        include("decisiontree.jl")
    end
end
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost


@setup_workload begin
    @compile_workload begin
        include("ensemble.jl")
    end
end
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

@setup_workload begin
    @compile_workload begin
        include("crossvalidator.jl")
    end
end
using .CrossValidators
export crossvalidate, pipe_performance

@setup_workload begin
    @compile_workload begin
        include("naremover.jl")
    end
end
using .NARemovers
export NARemover

@setup_workload begin
    @compile_workload begin
        include("pipelines.jl")
    end
end
using .Pipelines
export @pipeline, @pipelinex, @pipelinez
export |>, +, |, *, >>
export Pipeline, ComboPipeline

end # module
