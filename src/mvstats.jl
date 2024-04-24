module MultivarStats
using MultivariateStats
using MultivariateStats:NonlinearDimensionalityReduction
using CSV

const NDR=NonlinearDimensionalityReduction

using DataFrames: DataFrame
using Random
using AMLPipelineBase

using ..AbsTypes
using ..BaseFilters
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export testpca
export JlPreprocessor

const preprocessor_dict = Dict(
  "PCA"  => MultivariateStats.PCA,
  "KPCA" => MultivariateStats.KernelPCA,
  "PPCA" => MultivariateStats.PPCA,
  "FA"   => MultivariateStats.FactorAnalysis
)

"""
    JlPreprocessor(
       Dict(
         :name => "jlPCA",
         :k => 1
       )
    )
Returns PCA.

Implements `fit!` and `transform`.
"""
mutable struct JlPreprocessor <: Transformer
    name::String
    model::Dict{Symbol,Any}

    function JlPreprocessor(args::Dict = Dict{Symbol, Any}())
        default_args = Dict{Symbol,Any}(
           :name => "jlprep",
           :preprocessor => "PCA",
           :impl_args => Dict()
        )
        cargs=nested_dict_merge(default_args,args)
        cargs[:name] = cargs[:name]*"_"*randstring(3)
        prep = cargs[:preprocessor]
        if !(prep in keys(preprocessor_dict))
            println("$prep is not supported.")
            println()
            jlpreprocessors()
            error("Argument keyword error")
        end
        new(cargs[:name],cargs)
    end
end

function JlPreprocessor(name::String,args::Dict)
  JlPreprocessor(Dict(:preprocessor=>name,:name => name, args...))
end

function JlPreprocessor(name::String;opt...)
    JlPreprocessor(Dict(:preprocessor=>name,:name=>name,:impl_args=>Dict(pairs(opt))))
end

function jlpreprocessors()
  processors = keys(preprocessor_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: JlPreprocessor(name::String, args::Dict=Dict())")
  println("where *name* can be one of:")
  println()
  [print(processor," ") for processor in processors]
  println()
  println()
  println("and *args* are the corresponding preprocessor's initial parameters.")
end


function fit!(myprep::JlPreprocessor, x::DataFrame, y::Vector=[])::Nothing
    xtr = Matrix(x)'
    pargs = myprep.model[:impl_args]
    prepname = myprep.model[:preprocessor]
    obj = preprocessor_dict[prepname]
    myprep.model[:jlpreprocessor] = MultivariateStats.fit(obj,xtr;pargs...)
    return nothing
end

function transform!(myprep::JlPreprocessor, x::DataFrame)::DataFrame
    xtr = Matrix(x)'
    obj = myprep.model[:jlpreprocessor]
    MultivariateStats.predict(obj,xtr) |> x->DataFrame(x',:auto)
end

#"""
#    JlICA(
#       Dict(
#         :name => "JlICA",
#         :k => 1
#       )
#    )
#Returns PCA.
#
#Implements `fit!` and `transform`.
#"""
#mutable struct JlICA <: Transformer
#    name::String
#    model::Dict{Symbol,Any}
#
#    function JlICA(args::Dict = Dict{Symbol, Any}())
#        default_args = Dict{Symbol,Any}(
#           :name => "JlICA",
#           :k => 1,
#           :impl_args => Dict()
#        )
#        cargs=nested_dict_merge(default_args,args)
#        cargs[:name] = cargs[:name]*"_"*randstring(3)
#        new(cargs[:name],cargs)
#    end
#end
#
#function JlICA(name::String,args::Dict)
#  JlICA(Dict(:name => name, args...))
#end
#
#function JlICA(name::String;opt...)
#    JlICA(Dict(:name=>name,:impl_args=>Dict(pairs(opt))))
#end
#
#function fit!(myica::JlICA, x::DataFrame, y::Vector=[])::Nothing
#    xtr = Matrix(x)'
#    pargs = myica.model[:impl_args]
#    k = myica.model[:k]
#    myica.model[:preprocessor] = MultivariateStats.fit(ICA,xtr,k;pargs...)
#    return nothing
#end
#
#function transform!(myica::JlICA, x::DataFrame)::DataFrame
#    xtr = Matrix(x)'
#    preprocessor=myica.model[:preprocessor]
#    MultivariateStats.predict(preprocessor,xtr) |> x->DataFrame(x',:auto)
#end

function testpca()
    println("testpca")
    iris=CSV.File(joinpath(dirname(pathof(AMLPipelineBase)),"../data/iris.csv")) |> DataFrame
    features = iris[:,1:4] |> DataFrame
    for prep in keys(preprocessor_dict)
        println("model:",prep)
        model = JlPreprocessor(prep;maxoutdim=2)
        fit!(model,features)
        r=transform!(model,features) 
        println(r[1,:])
    end
end

end
