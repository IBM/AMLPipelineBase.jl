module TestFeatureSelectors

using Random
using Test
using CSV
using Statistics
using AMLPipelineBase
using DataFrames: DataFrame

function iris_test()
    X = getiris()
    catfeat = FeatureSelector([5])
    numfeat = FeatureSelector([1,2,3,4])
    catf = CatFeatureSelector()
    numf = NumFeatureSelector()
    @test (fit_transform!(catfeat,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform(catfeat,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(numfeat,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @test (fit_transform(numfeat,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @test (fit_transform!(catf,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform(catf,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(numf,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @test (fit_transform(numf,X) .== X[:,1:4]) |> Matrix |> sum == 600
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator()
    res = fit_transform!(catnum,catnumdata)
    @test infer_eltype(catnumdata[:,[2,4,6]]) <: Number
    @test infer_eltype(res[:,[2,4,6]]) <: String
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator(0)
    res = fit_transform!(catnum,catnumdata)
    @test eltype(res[:,6]) <: Number
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    res1 = fit_transform(catnum,catnumdata)
    @test eltype(res1[:,6]) <: Number
end
@testset "Feature Selectors: Iris" begin
    Random.seed!(123)
    iris_test()
end

function diabetes_test()
    Random.seed!(123)
    diabetesdf = CSV.File(joinpath(dirname(pathof(AMLPipelineBase)),"../data/diabetes.csv")) |> DataFrame
    X = diabetesdf[:,1:end-1]
    Y = diabetesdf[:,end] |> Vector
    acc(X,Y)=score(:accuracy,X,Y)
    dt     = PrunedTree()
    ada    = Adaboost()
    rf     = RandomForest()
    ohe    = OneHotEncoder()
    catf   = CatFeatureSelector()
    numf   = NumFeatureSelector()
    disc = CatNumDiscriminator(20)
    pl = @pipeline disc |> ( (catf |> ohe)) |> rf
    @test crossvalidate(pl,X,Y,acc,2,false).mean > 0.60
    @test score(:accuracy, fit_transform(pl,X,Y),Y) > 50.0
    @test score(:accuracy, fit_transform!(pl,X,Y),Y) > 50.0
end
@testset "Feature Selectors: Diabetes" begin
    Random.seed!(123)
    diabetes_test()
end

end
