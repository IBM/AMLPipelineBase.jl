module TestAMLPipeline
using Test
include("test_featureselector.jl")
include("test_util.jl")
include("test_baselinemodel.jl")
include("test_basefilter.jl")
include("test_decisiontree.jl")
include("test_ensemble.jl")
include("test_crossvalidator.jl")
include("test_naremover.jl")
include("test_pipeline.jl")
end
