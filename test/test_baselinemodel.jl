module TestBaseline

using Random
using Test
using DataFrames: nrow
using AMLPipelineBase

function test_baseline()
    Random.seed!(123)
    iris=getiris()
    features=iris[:,1:4]
    labels=iris[:,5] |> collect
    bl = Baseline()
    fit!(bl,features,labels)
	 @test bl.model[:choice] == "setosa"
    @test sum(transform!(bl,features) .== repeat(["setosa"],nrow(iris))) == nrow(iris)
    @test sum(fit_transform(bl,features,labels) .== repeat(["setosa"],nrow(iris))) == nrow(iris)
    idy = Identity()
    fit!(idy,features,labels)
    @test (transform!(idy,features) .== features) |> Matrix |> sum == 150*4
    @test (fit_transform(idy,features) .== features) |> Matrix |> sum == 150*4
end
@testset "Baseline Tests" begin
  test_baseline()
end

end
