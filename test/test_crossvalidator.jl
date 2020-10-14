module TestCrossValidator

using Test
using Random
using AMLPBase

function test_crossvalidator()
  racc = 50.0
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> Vector
  rf = RandomForest()
  @test crossvalidate(rf,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  ppl1 = Pipeline([RandomForest()])
  @test crossvalidate(ppl1,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  ohe = OneHotEncoder()
  ppl2 = @pipeline ohe |> rf
  @test crossvalidate(ppl2,X,Y,acc,10,false).mean > racc
end
@testset "CrossValidator" begin
  test_crossvalidator()
end


end
