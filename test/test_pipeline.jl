module TestPipeline

using Random
using Test
using AMLPipelineBase
using AMLPipelineBase.Pipelines
using AMLPipelineBase.DecisionTreeLearners
using AMLPipelineBase.CrossValidators
using AMLPipelineBase.Utils

const data     = getiris()
const features = data[:,1:4]
const X        = data[:,1:5]
const Y        = data[:,5] |> Vector
X[!,5]         = X[!,5] .|> string

const ohe = OneHotEncoder()
const noop = Identity()
const rf  = RandomForest()
const ada = Adaboost()
const pt  = PrunedTree()

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
  res1 = fit_transform!(combo1,X)
  res2 = fit_transform!(combo2,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  res1 = fit_transform(combo1,X)
  res2 = fit_transform(combo2,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res4) |> Matrix |> sum == 2100
  pcombo1 = @pipeline ohe + ohe
  pres1 = fit_transform!(pcombo1,X)
  @test (pres1 .== res1) |> Matrix |> sum == 2100
  combo3 = ohe + ohe
  combo5 = linear1 + linear2
  res5 = fit_transform!(combo3,X)
  @test (res5 .== res2) |> Matrix |> sum == 2100
  res6 = fit_transform!(combo5,X)
  @test (res6 .== res2) |> Matrix |> sum == 2100
  res7 = fit_transform(combo5,X)
  @test (res6 .== res7) |> Matrix |> sum == 2100
end
@testset "Pipelines" begin
  Random.seed!(123)
  test_pipeline()
end

acc(X,Y) = score(:accuracy,X,Y)

function test_sympipeline()
  pcombo5 = @pipeline :((ohe + noop) |> (ada * rf * pt))
  pcombo6 = (ohe + noop) |> (ada * rf * pt)
  @test crossvalidate(pcombo5,X,Y,acc,5,false).mean >= 0.90
  @test crossvalidate(pcombo6,X,Y,acc,5,false).mean >= 0.90
  expr = :((ohe + noop) |> (ada * rf * pt))
  processexpr!(expr.args)
  @test crossvalidate(eval(expr),X,Y,acc,5,false).mean >= 0.90
  pcombo6 = sympipeline(expr) |> eval
  @test crossvalidate(pcombo6,X,Y,acc,5,false).mean >= 0.90
  pcombo7 = (@pipelinez expr) |> eval
  @test crossvalidate(pcombo7,X,Y,acc,5,false).mean >= 0.90
end
@testset "Symbolic Pipeline: Global Scope" begin
  Random.seed!(123)
  test_sympipeline()
end

function test_pipeline()
  # test symbolic pipeline expression 
  pcombo2 = @pipeline ohe + noop
  pcombo3 = ohe + noop
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 158
  @test fit_transform(pcombo2,features) |> Matrix |> size |> collect |> sum == 158
  @test fit_transform!(pcombo3,features) |> Matrix |> size |> collect |> sum == 158
  @test fit_transform(pcombo3,features) |> Matrix |> size |> collect |> sum == 158
  pcombo4 = @pipeline ohe + noop |> rf
  pcombo5 = (ohe + noop) |> rf
  @test crossvalidate(pcombo4,X,Y,acc,5,false).mean >= 0.90
  @test crossvalidate(pcombo5,X,Y,acc,5,false).mean >= 0.90
end
@testset "Symbolic Pipeline: Local Scope" begin
  Random.seed!(123)
  test_pipeline()
end

function test_split()
   dt1 = train_test_split(features,Y)
   perf1 = pipe_performance(rf,(x,y)->score(:accuracy,x,y), dt1.trX,dt1.trY,dt1.tstX,dt1.tstY)
   @test perf1 > 50.0
   dt2 = train_test_split(data[:,1:3],Vector(data[:,4]))
   perf2 = pipe_performance(rf,(x,y)->score(:rmse,x,y), dt2.trX,dt2.trY,dt2.tstX,dt2.tstY)
   @test perf2 < 0.50
end
@testset "Performance and train/test split" begin
  Random.seed!(123)
  test_split()
end

end
