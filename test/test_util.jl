module TestUtil

using Test
using AMLPipelineBase
using AMLPipelineBase.Utils
using DataFrames
using Random


function test_utils()
  data = getiris()

  catnum=find_catnum_columns(data)
  @test catnum == ([5],[1,2,3,4])

  (r,l)=holdout(nrow(data),0.9)
  @test length.((r,l)) == (15,135)

  kf=kfold(nrow(data),10)
  @test size.(kf) |> unique == [(135,)]

  Y=data.Species |> collect
  @test score(:accuracy,Y,Y) == 100.0
  @test score(:accuracy,Y,reverse(Y)) |> round == 33.0

  @test infer_eltype.(eachcol(data)) == [Float64,Float64,Float64,Float64,String]

  dc = Dict(Dict(:a=>:c,:d=>Dict(:e=>:f)))
  @test nested_dict_to_tuples(dc) == Set(Any[([:d, :e], :f), ([:a], :c)]) 

  dx = deepcopy(dc)
  nested_dict_set!(dx,[:d,:e],:h)
  @test dx[:d][:e] == :h

  @test nested_dict_merge(dx,dc) == dc
  @test mergedict(dc,dx) == dx

  x = [2,200,332,201,missing,50,missing,6,missing,7,8,109]
  @test skipmean(x) |> round == 102.0 
  @test skipmedian(x) == 50.0
  @test skipstd(x) |> round == 118.0
end
@testset "Utils" begin
  test_utils()
end


function test_metric()
   data = getiris()
   X=data[:,2:end]
   Y=data[:,1] |> collect
   rf = RandomForest()
   catf = CatFeatureSelector()
   numf = NumFeatureSelector()
   ohe = OneHotEncoder()

   pl = @pipeline (catf |> ohe) + (numf) |> rf

   rmse(X,Y) = score(:rmse,X,Y)
   @test crossvalidate(pl,X,Y,rmse,10,false).mean < 0.5
   @test crossvalidate(pl,X,Y;metric=rmse,verbose=false).mean < 0.5

   mse(X,Y) = score(:mse,X,Y)
   @test crossvalidate(pl,X,Y,mse,10,false).mean < 0.2
   @test crossvalidate(pl,X,Y;metric=mse,verbose=false).mean < 0.2

   mae(X,Y) = score(:mae,X,Y)
   @test crossvalidate(pl,X,Y,mae,10,false).mean < 0.2
   @test crossvalidate(pl,X,Y;metric=mae,verbose=false).mean < 0.2

   rmse(X,Y) = score(:rmse,X,Y)
   @test crossvalidate(pl,X,Y,mae,10,false).mean < 0.2
   @test crossvalidate(pl,X,Y;metric=mae,verbose=false).mean < 0.2
end
@testset "Utils Metrics" begin
  test_metric()
end


end
