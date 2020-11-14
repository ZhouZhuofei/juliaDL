### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 695f3a2c-2640-11eb-3c75-93d1ed0e053a
html"""
<center><img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia-logo-mask.png" height="50" /></center>
"""

# ╔═╡ bf750d4c-25b9-11eb-17ca-dd85d876688d
md"""
# Statistics With Julia Example
"""

# ╔═╡ 30719ba6-25d4-11eb-38b9-11df9c03dfa8
md"""
## Example
### Bubble Sort

A basic sorting algorithm.

Takes an input array, index $1,...,n$, then sorts the elemnts smallesr to largest by allowing the larger elemnts, or "bubble".

```julia

function bubbleSort!(a)
    n = length(a)
    for i in 1:n-1
        for j in 1: n-i
            if a[j] > a[j+1]
                a[j], a[j+1] = a[j+1], a[j]
            end
        end
    end
    return a
end
```
"""

# ╔═╡ ac77aa56-2640-11eb-3a03-fdec7724034b
md"""
### Roots of a Polynomial

Consider the polynomial

$f(x) = a_nx^n + a_{n-1}x^{n-1} + ...+ a_1x + a_0$

with real-valued coefficients $a_0, ...a_n$.find all $x$ values that solve the equation $f(x)=0$.

```julia
using Roots
function polynomialGenerator(a...)
    n = length(a) - 1
    poly = function(x)
				return sum([a[i+1] * x^i for i in 0:n])
			end
	return poly
end

polynomial = polynomialGenerator(1,3,-10)
zeroVals = find_zeros(polynomial,-10,10)
println("Zeros of the function f(x): ", zeroVals)
```

"""

# ╔═╡ d15f3bc6-2641-11eb-2160-87492eb2595f
md"""
### Markov Chain

1. By raising the Matrix $P$ to a high power, the limiting distribution is obtained in any row.$\pi_i = lim_{n \to \infty}[P^n]_{i,j}$ for any index, $j$.

```julia
P = [0.5 0.4 0.1;
	 0.3 0.2 0.5;
	 0.5 0.3 0.2]

res1 = (P^100)[1, :]
```

2. We solve the linear systerm of equations,$\pi P = \pi$ and $\sum_{i=1}^{3} \pi_i = 1$

```julia
A = vcat((P' - I)[1:2, :], ones(3)')
b = [0 0 1]'
res2 = A\b
```
"""

# ╔═╡ 7b005a6e-2644-11eb-3f9a-0523247ad483
md"""
### Web,JSON and String Process

The **JSON** format uses '{}' characters to enclose a hierarchical nested structure of key value pairs.
```julia
using HTTP, JSON
data = HTTP.request("GET", "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")
shakespeare = String(data.body)
shakespeareWords = split(shakespeare)

jsonWords = HTTP.request("GET", "https://raw.githubusercontent.com/"*
"h-Klok/StatsWithJuliaBook/master/1_chapter/jsonCode.json")
parsedJsonDict = JSON.parse(String(jsonWords.body))

keywords = Array{STring}(parsedJsonDict["words"])

numberToSHow = parsedJsonDict["numToShow"])

wordCount = Dict([(x, count(w -> lowrcase(w) == lowercase(x), shakespeareWords)) for x in keywords])
sortedWordCount = sort(collect(wordCount),by=last,rev=true)
sortedWordCount[1:numberToShow]

```


"""

# ╔═╡ 5f3a8406-2646-11eb-050b-47d517ceadf8
md"""
### Basic Plotting

you can visit the website[http://docs.juliaplots.org/](http://docs.juliaplots.org/), to get more information about plots

```julia
using Plots, LaTeXStrings, Measures
f(x,y) = x^2 + y^2
f0(x) = f(x,0)
f2(x) = f(x,2)
xVals, yVals = -5:0.1:5 , -5:0.1:5
plot(xVals, [f0.(xVals), f2.(xVals)],c=[:blue :red], xlims=(-5,5), legend=:top,
    ylims=(-5,25), ylabel=L"f(x,\cdot)", label=[L"f(x,0)" L"f(x,2)"])
p1 = annotate!(0, -0.2, text("(0,0) The minimum\n of f(x,0)", :left, :top, 10))
z=[f(x,y) for y in yVals, x in xVals]
p2 = surface(xVals, yVals, z, c=cgrad([:blue, :red]),legend=:none,
    ylabel="y", zlabel=L"f(x,y)")
M = z[1:10,1:10]
p3 = heatmap(M, c=cgrad([:blue, :red]), yflip=true, ylabel="y",
    xticks=([1:10;], xVals), yticks=([1:10;], yVals))
plot(p1, p2, p3, layout=(3,1), size=(1000,1200), xlabel="x", margin=5mm)
```

"""

# ╔═╡ Cell order:
# ╟─695f3a2c-2640-11eb-3c75-93d1ed0e053a
# ╠═bf750d4c-25b9-11eb-17ca-dd85d876688d
# ╟─30719ba6-25d4-11eb-38b9-11df9c03dfa8
# ╠═ac77aa56-2640-11eb-3a03-fdec7724034b
# ╠═d15f3bc6-2641-11eb-2160-87492eb2595f
# ╟─7b005a6e-2644-11eb-3f9a-0523247ad483
# ╟─5f3a8406-2646-11eb-050b-47d517ceadf8
