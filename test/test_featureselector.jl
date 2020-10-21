module TestFeatureSelectors

using Random
using Test
using CSV
using Statistics
using DataFrames
using AMLPipelineBase

function iris_test()
    X = getiris()
    catfeat = FeatureSelector([5])
    numfeat = FeatureSelector([1,2,3,4])
    catf = CatFeatureSelector()
    numf = NumFeatureSelector()
    @test (fit_transform!(catfeat,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(numfeat,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @test (fit_transform!(catf,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(numf,X) .== X[:,1:4]) |> Matrix |> sum == 600
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
    catnum = CatNumDiscriminator(5)
    #ppp=@pipeline catnum |> ((catf |> OneHotEncoder()) + (numf |> Normalizer(Dict(:method=>:pca))))
    #res=fit_transform!(ppp,catnumdata)
    #@test ncol(res) == 11
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

    #zscore = Normalizer(Dict(:model =>:zscore))
    #unitr  = Normalizer(Dict(:model =>:unitrange))
    #fa     = Normalizer(Dict(:model =>:fa))
    #pca    = Normalizer(Dict(:model =>:pca))
    dt     = PrunedTree()
    ada    = Adaboost()
    rf     = RandomForest()
    ohe    = OneHotEncoder()
    catf   = CatFeatureSelector()
    numf   = NumFeatureSelector()

    #disc = CatNumDiscriminator(0)
    #pl = @pipeline disc |> ((numf |>  pca) + (catf |> ohe)) |> rf
    #@test crossvalidate(pl,X,Y,acc,10,false).mean > 0.60

    #pl = @pipeline disc |> ((numf |> zscore |>  pca) + (catf |> ohe)) |> ada
    #@test crossvalidate(pl,X,Y,acc,10,false).mean > 0.60

    #pl = @pipeline disc |> ((numf |> unitr |>  fa) + (catf |> ohe)) |> dt
    #@test crossvalidate(pl,X,Y,acc,10,false).mean > 0.60

    disc = CatNumDiscriminator(20)
    pl = @pipeline disc |> ( (catf |> ohe)) |> rf
    @test crossvalidate(pl,X,Y,acc,2,false).mean > 0.60

    #disc = CatNumDiscriminator(50)
    #pl = @pipeline disc |> ((numf |> zscore |>  pca) + (catf |> ohe)) |> rf
    #@test crossvalidate(pl,X,Y,acc,2,false).mean > 0.60
end
@testset "Feature Selectors: Diabetes" begin
    Random.seed!(123)
    diabetes_test()
end

end
