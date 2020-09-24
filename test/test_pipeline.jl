module TestPipeline

using Random
using Test
using AMLPBase
using AMLPBase.Pipelines
using AMLPBase.DecisionTreeLearners
using AMLPBase.CrossValidators
using AMLPBase.Utils

global const data     = getiris()
global const features = data[:,1:4]
global const X        = data[:,1:5]
const Y               = data[:,5] |> Vector
X[!,5]                = X[!,5] .|> string

global const ohe = OneHotEncoder()
global const noop = Identity()
global const rf  = RandomForest()
global const ada = Adaboost()
global const pt  = PrunedTree()

function test_pipeline()
  # test initialization of types
  ohe = OneHotEncoder()
  linear1 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  linear2 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  combo1 = ComboPipeline(Dict(:machines=>[ohe,ohe]))
  combo2 = ComboPipeline(Dict(:machines=>[linear1,linear2]))
  linear1 = Pipeline([ohe])
  linear2 = Pipeline([ohe])
  combo1 = ComboPipeline([ohe,ohe])
  combo2 = ComboPipeline([linear1,linear2])
  # test fit/transform workflow
  fit!(combo1,X)
  res1=transform!(combo1,X)
  res2=fit_transform!(combo1,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res4) |> Matrix |> sum == 2100
  pcombo1 = @pipeline ohe + ohe
  pres1 = fit_transform!(pcombo1,X)
  @test (pres1 .== res1) |> Matrix |> sum == 2100
end
@testset "Pipelines" begin
  Random.seed!(123)
  test_pipeline()
end

acc(X,Y) = score(:accuracy,X,Y)

function test_sympipeline()
  pcombo5 = @pipeline :((ohe + noop) |> (ada * rf * pt))
  @test crossvalidate(pcombo5,X,Y,acc).mean >= 0.90
  expr = :((ohe + noop) |> (ada * rf * pt))
  processexpr!(expr.args)
  @test crossvalidate(eval(expr),X,Y,acc).mean >= 0.90
  pcombo6 = sympipeline(expr) |> eval
  @test crossvalidate(pcombo6,X,Y,acc).mean >= 0.90
  pcombo7 = (@pipelinez expr) |> eval
  @test crossvalidate(pcombo7,X,Y,acc).mean >= 0.90
end
@testset "Symbolic Pipeline: Global Scope" begin
  Random.seed!(123)
  test_sympipeline()
end

function test_pipeline()
  # test symbolic pipeline expression 
  pcombo2 = @pipeline ohe + noop
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 158
  pcombo2 = @pipeline ohe + noop |> rf
  @test crossvalidate(pcombo2,X,Y,acc).mean >= 0.90
end
@testset "Symbolic Pipeline: Local Scope" begin
  Random.seed!(123)
  test_pipeline()
end

end
