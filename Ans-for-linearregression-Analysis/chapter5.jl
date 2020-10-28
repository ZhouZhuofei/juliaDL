### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 6a66d77c-17a4-11eb-0061-f3703df5864c
begin
	using DataFrames
	using CSV
	using Plots
	using GLM
	using Statistics
	using StatsModels
	using StatsPlots
	using Latexify
	using Distributions
end

# ╔═╡ c98883c0-1794-11eb-2d9b-4138e0d901d8
md"""_Test 5_

Submission by: _**Zhuofei, Zhou**_(fregot@icloud.com)

_julia version: 1.5.2_
"""

# ╔═╡ fab7c734-1793-11eb-2c22-3f7a797bd617
md"# Modification of model inapplicability transformation and weighting"

# ╔═╡ 30b3cf96-17a0-11eb-3e5b-cd68a544c808
md"here,import some packages:"

# ╔═╡ f71e2f66-17ac-11eb-18cc-e1ae7832fecc
md"some functions:"

# ╔═╡ ff14ac7a-17ac-11eb-2385-c3565c4ed946
function calculate_one_R2(x::Vector,y::Vector)
    m = size(x)[1]
    X = Float64[x ones(m)]
    β = inv(X'*X)*X'*y
    SSc = y - X*β
    dfc = m - 2
    MSc = SSc/dfc
    e = SSc
    SSh = X*β - mean(y)*ones(m)
    SSH = SSh'*SSh
    SSC = SSc'*SSc
    SST = SSC + SSH
    R = SSH/SST*100
    return R
end

# ╔═╡ 08c6a4ba-17af-11eb-35ff-2f7daa814291

function normal_one_probility(x, y)
    m = size(x)[1]
    X = Float64[x ones(m)]
    β = inv(X'*X)*X'*y
    SSc = y - X*β
    SSC = SSc'*SSc
    dfc = m - 2
    MSc = SSC/dfc
    e = X*β - y
    H = X*inv(X'*X)*X'
    Sᵢ = zeros(m)
    tᵢ = zeros(m)
    p = zeros(m)
    rᵢ = zeros(m)
    press = zeros(m)
    for i  in 1:m
        Sᵢ[i] = ((m-1)*MSc - e[i]/(1 - H[i,i]))/(m - 2)
        tᵢ[i] = e[i]/sqrt(Sᵢ[i]^2*(1 - H[i, i]))
        rᵢ[i] = e[i]/(MSc*(1 - H[i, i]))
        p[i] = (i - 0.5)/m
        press[i] = (e[i]/(1-H[i, i]))^2
    end
    res = DataFrame(y_pre = X*β, e = SSc, Sᵢ = Sᵢ,rᵢ = rᵢ, Tᵢ = tᵢ, p = p, PRESS = press)
    #println(res)
    return res
end

# ╔═╡ ec6c2cc6-17b4-11eb-0b5e-a90d68d8c6b0
function NormalProbabilityPlot(data)
	mu=mean(data)
	sig=std(data)
	n=length(data)
	p=[(i -0.5)/n for i in 1:n]
	x=quantile(Normal(), p)
	y=sort([(i-mu)/sig for i in data])
	scatter(x, y)
end

# ╔═╡ b8ce3ebc-18f0-11eb-3dd5-3f799465f5ec
correct(text) = Markdown.MD(Markdown.Admonition("correct", "Ans", [text]))

# ╔═╡ 3ef4c8bc-18f4-11eb-3f4a-7179d5e3f071
note(text) = Markdown.MD(Markdown.Admonition("note", "Note", [text]))

# ╔═╡ c7bd8dae-1794-11eb-145e-4938a72ebf82
md"""

## 5.1
Byers and Williams investigated the effect of temperture(regression variable) on the viscosity (response variable) of toluene-tetrahydronaphthalene mixtures.
> Byers and williams研究了温度（回归变量）对甲苯-四氢化萘混合物黏度（响应变量）的影响
1. make a scatter diagram to see if the linear relationship is reasonable.
2. fit the linear model, calculate the President messurement, and make the residual graph. What conclusions can be drawn about the applicable performance of the model
3. The basic priciples of physical chemistry indicate that viscosity is an exponential function of tempterature, redo 2 using transformation.

Data:
"""

# ╔═╡ 20e05c08-17a0-11eb-17d2-c3862d4f222c
begin
	t = [24.9, 35.0, 44.9, 55.1, 65.2, 75.2, 85.2, 95.2]
	y = [1.133, 0.9722, 0.8532, 0.7550, 0.6723, 0.6021, 0.5420, 0.5074]
	data = DataFrame(t=t, y=y)
end

# ╔═╡ 8e0208d6-17a5-11eb-3571-8b4708ecc3de
md"Solutions:
"

# ╔═╡ a9b16f9a-17a5-11eb-1c8b-7b6b4e5bdc68
scatter(t, y, xlabel="t/temperture", ylabel="Pa⋅S", leg=:none)

# ╔═╡ 2d8d245a-17ac-11eb-0d4c-dfb10eeb06e6
ols1 = lm(@formula(y ~ t), data)

# ╔═╡ 26ebed02-17ae-11eb-37f3-8b945ab3fed9
begin
	e = y - predict(ols1)
	scatter(predict(ols1),e)
end

# ╔═╡ 6fff184e-17ad-11eb-3801-0b72654cb699
md"""
$ R^2 = $
"""

# ╔═╡ 4e4dc68c-17ad-11eb-09f1-4fad10ca45e4
latexify(calculate_one_R2(t,y); fmt = "%.2f")

# ╔═╡ 8dbe1f26-17ae-11eb-0f57-c77e8253bb96
correct(md"the residual plot shows a nonlinear pattern and normality is violated.")

# ╔═╡ 908a0072-17af-11eb-07fa-9938001bde9b
res = normal_one_probility(t, y)

# ╔═╡ f650b660-17b1-11eb-065c-f35472d4cf72
res.Tᵢ

# ╔═╡ bae36286-17b0-11eb-0740-53d8bb54e0cc
scatter(t,[log.(y),y])

# ╔═╡ d53f976a-17b0-11eb-147b-c193db7df691
res1 = normal_one_probility(t, log.(y))

# ╔═╡ 03f6bdf4-17b1-11eb-15bb-6d1d0ae5be04
scatter(res1.y_pre, res1.rᵢ)

# ╔═╡ 9f737ec4-17b4-11eb-1384-351e209c6cb8

NormalProbabilityPlot(res.rᵢ)

# ╔═╡ a8b71038-17b2-11eb-3ac4-d5ff9fffaa11
correct(md"There is slight improvement in the model")

# ╔═╡ fe82916a-18f0-11eb-0f29-2fa6d9326c13
html"<br></br>"

# ╔═╡ 095bae32-18f1-11eb-3ff7-cfdc4125d745
md"""
## 5.2

The vapor pressure of water at different temperatures.

1. make a scatter diagram to see if the linear relationship is reasonable.

2. fitting linear model, and calculate President messurement, and make the residual graph. What conclusions can be drawn about the applicable performance of the model.

3.  The clausius-Clapeyron equation in physical chemistry states: $ln(p_c) \sim -\frac1T$, based on the information, do some appropriate transformission.
"""

# ╔═╡ 8f80cc44-18f2-11eb-12d4-5727ca38789d
#data
begin
	mmHg = [4.6, 9.2, 17.5, 31.8, 55.3, 92.5, 149.4, 233.7, 355.1, 525.8, 760.0]
	data2 = DataFrame(t = [273, 283, 293, 303, 313, 323, 333, 343, 353, 363, 373], mmHg = mmHg)
end

# ╔═╡ 029f2540-18f3-11eb-3523-f35f3e2ac61f
scatter(data2.t, data2.mmHg, leg=:none, xlabel="temperatures", ylabel="mmHg")

# ╔═╡ 383465bc-18f3-11eb-09b6-51709b7a4053
correct(md"absolutely, linear relationship is unreasonable.")

# ╔═╡ 05a7a704-18f4-11eb-3bf0-3de804963e50
ols2= lm(@formula(mmHg ~ t), data2)

# ╔═╡ 595e4ea4-18f3-11eb-3e10-8f972926492d
begin
	res2 = normal_one_probility(data2.t, data2.mmHg)
	scatter(res2.y_pre, res2.e, title="residuals plot", leg=:none)
end

# ╔═╡ c14304bc-18f3-11eb-087a-4fe3c1c9f1ce
correct(md"Direct use of linear model isn't a good way.")

# ╔═╡ f27b2e74-18f3-11eb-1978-73bd712aec51
note(md"we are kown ``ln(p) \sim -\frac1T``,so we can do like:``y' = ln(mmHg), x'=\frac1T``")

# ╔═╡ a5642522-18f4-11eb-1c36-b3e1e39249b1
begin
	ŷ₂ = log.(data2.mmHg)
	x̂₂ = 1 ./ data2.t
	data2.y_t = ŷ₂
	data2.x_t = x̂₂
	ols22 = lm(@formula(y_t ~ x_t), data2)
end

# ╔═╡ 5f58147a-18f5-11eb-2004-75b188ce8a86
begin
	res22 = normal_one_probility(data2.x_t, data2.y_t)
	scatter(res22.y_pre, res22.e, title="after transmission residuals plot", leg=:none)
end

# ╔═╡ 89c38532-18f5-11eb-2b17-0f367a4a51fd
scatter(data2.x_t, data2.y_t, leg=:none)

# ╔═╡ ae63ac1e-18f5-11eb-2790-c5d647e2c52a
correct(md"the function: ``ln(mmHg) = -5200.76 \cdot \frac1T + 20.6074``")

# ╔═╡ dbe4f300-18f5-11eb-3751-2f7a7997a159
html"<br></br>"

# ╔═╡ Cell order:
# ╟─c98883c0-1794-11eb-2d9b-4138e0d901d8
# ╟─fab7c734-1793-11eb-2c22-3f7a797bd617
# ╟─30b3cf96-17a0-11eb-3e5b-cd68a544c808
# ╟─6a66d77c-17a4-11eb-0061-f3703df5864c
# ╟─f71e2f66-17ac-11eb-18cc-e1ae7832fecc
# ╟─ff14ac7a-17ac-11eb-2385-c3565c4ed946
# ╟─08c6a4ba-17af-11eb-35ff-2f7daa814291
# ╟─ec6c2cc6-17b4-11eb-0b5e-a90d68d8c6b0
# ╠═b8ce3ebc-18f0-11eb-3dd5-3f799465f5ec
# ╟─3ef4c8bc-18f4-11eb-3f4a-7179d5e3f071
# ╟─c7bd8dae-1794-11eb-145e-4938a72ebf82
# ╟─20e05c08-17a0-11eb-17d2-c3862d4f222c
# ╟─8e0208d6-17a5-11eb-3571-8b4708ecc3de
# ╟─a9b16f9a-17a5-11eb-1c8b-7b6b4e5bdc68
# ╟─2d8d245a-17ac-11eb-0d4c-dfb10eeb06e6
# ╟─26ebed02-17ae-11eb-37f3-8b945ab3fed9
# ╟─6fff184e-17ad-11eb-3801-0b72654cb699
# ╟─4e4dc68c-17ad-11eb-09f1-4fad10ca45e4
# ╟─8dbe1f26-17ae-11eb-0f57-c77e8253bb96
# ╠═908a0072-17af-11eb-07fa-9938001bde9b
# ╠═f650b660-17b1-11eb-065c-f35472d4cf72
# ╠═bae36286-17b0-11eb-0740-53d8bb54e0cc
# ╠═d53f976a-17b0-11eb-147b-c193db7df691
# ╠═03f6bdf4-17b1-11eb-15bb-6d1d0ae5be04
# ╠═9f737ec4-17b4-11eb-1384-351e209c6cb8
# ╟─a8b71038-17b2-11eb-3ac4-d5ff9fffaa11
# ╟─fe82916a-18f0-11eb-0f29-2fa6d9326c13
# ╟─095bae32-18f1-11eb-3ff7-cfdc4125d745
# ╟─8f80cc44-18f2-11eb-12d4-5727ca38789d
# ╟─029f2540-18f3-11eb-3523-f35f3e2ac61f
# ╟─383465bc-18f3-11eb-09b6-51709b7a4053
# ╠═05a7a704-18f4-11eb-3bf0-3de804963e50
# ╠═595e4ea4-18f3-11eb-3e10-8f972926492d
# ╟─c14304bc-18f3-11eb-087a-4fe3c1c9f1ce
# ╟─f27b2e74-18f3-11eb-1978-73bd712aec51
# ╟─a5642522-18f4-11eb-1c36-b3e1e39249b1
# ╟─5f58147a-18f5-11eb-2004-75b188ce8a86
# ╟─89c38532-18f5-11eb-2b17-0f367a4a51fd
# ╟─ae63ac1e-18f5-11eb-2790-c5d647e2c52a
# ╟─dbe4f300-18f5-11eb-3751-2f7a7997a159
