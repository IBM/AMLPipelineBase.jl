module TestUtil

using Test
using AMLPipelineBase
using AMLPipelineBase.Utils
using DataFrames


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

end
