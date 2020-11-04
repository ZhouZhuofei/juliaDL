### A Pluto.jl notebook ###
# v0.12.6

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
	qqplot(x, y, qqline=:R, xlabel="Theoretical Quantiles", ylabel="Standardized Residuals", title="Normal Q-Q")
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
NormalProbabilityPlot(res.e)

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

# ╔═╡ b5499d0a-1929-11eb-346c-2fc9cda07625
md"""
## 5.3
The average number of living 🦠(bacteria) in canned food products and the number of minutes exposed to 300∘F.

1. make a scatter diagram to see if the linear relationsihp is reasionable.

2. fitting linear model, calculate President messurement, and make the residual graph. What conclusions can be drawn about the applicable performance of the model.

3. The suitable model transformation for this data is identified, and the data is fitted with the transformed model and the general test of model applicability is carried out.
"""


# ╔═╡ a85eb29c-192b-11eb-0d05-c1087fb1ee63
data3 = DataFrame(time=[1,2,3,4,5,6,7,8,9,10,11,12],number=[175,108,95,82,71,50,49,31,28,17,16,11])

# ╔═╡ f5672d3a-192b-11eb-271d-3f82c74c1fce
scatter(data3.time, data3.number, leg=:none, xlabel="expose time", ylabel="bacteria numbers")

# ╔═╡ 80c4ddf0-192c-11eb-02eb-6b8e5ed1ecde
correct(md"Direct use of linear model isn't a good way.")

# ╔═╡ 88253dec-192c-11eb-3fe4-d3c9f3eb9882
ols3 = lm(@formula(number ~ time), data3)

# ╔═╡ f1750ab0-192d-11eb-23c4-f9f82e4343e7
r2(ols3)

# ╔═╡ a94f5610-192c-11eb-387f-87f04e2d3774
begin
	res3 = normal_one_probility(data3.time, data3.number)
	scatter(res3.y_pre, res3.e, leg=:none, xlabel="predict y value", ylabel="residual")
end

# ╔═╡ b29e4abc-192e-11eb-23ce-c52e1b6c41e0
NormalProbabilityPlot(res3.e)

# ╔═╡ c0f3ad04-192f-11eb-166b-59f0ccd50ce1
note(md"we can find a obvious outliers point, if we delect the point.

here we choose two ways to do linear transformation.

- ``y=\beta_0 e ^{\beta_1 x}``
- ``y = \beta_0 + \beta_1 log x`` ")

# ╔═╡ 9ac087a0-1930-11eb-28e9-e7886e5816bc
scatter(data3.time, log.(data3.number))

# ╔═╡ b367c318-1930-11eb-0d0b-53498dd6eab0
scatter(log.(data3.time), data3.number)

# ╔═╡ aceb07f2-1933-11eb-122b-4185d4d50031
begin
	res33 = normal_one_probility(log.(data3[2:end, :].time), data3[2:end, :].number)
	scatter(res33.y_pre, res33.e, leg=:none)
end

# ╔═╡ 00a65f8a-1941-11eb-0ece-7beb73a126d6
correct(md"so, if we delect a outlier point, and do a ``x' = ln(x)``")

# ╔═╡ a74d3a02-1941-11eb-2192-510011d1f284
html"<br></br>"

# ╔═╡ dee284c2-1941-11eb-0dab-ab42595e7d56
md"""
## 5.4
Considering the data shown below, the scatter diagram is constructed and the appropriate form of the refression model is suggested.
"""

# ╔═╡ 0102859c-1943-11eb-332e-3f896f4114e5
data4 = DataFrame(x=[10,15,18,12,9,8,11,6], y=[0.17,0.13,0.09,0.15,0.20,0.21,0.18,0.24])

# ╔═╡ 32ab63f2-1943-11eb-2ece-03627d4a4e09
scatter(data4.x, data4.y, leg=:none)

# ╔═╡ 4c06b20c-1943-11eb-3191-21e1fb66dbc8
ols4 = lm(@formula(y ~ x), data4)

# ╔═╡ 69ff0ebc-1943-11eb-101b-43f5f6d7ce18
begin
	res4 = normal_one_probility(data4.x, data4.y)
	scatter(res4.y_pre, res4.e, xlabel="fitting value", ylabel="residual")
end

# ╔═╡ 9afdcdfa-1943-11eb-26e4-eb1268dfe551
correct(md"``y = -0.0121215 \cdot x + 0.306103``, but the residual plot does not look good. Taking the _natural log of x_ makes better model.")

# ╔═╡ 6e64f52e-1944-11eb-05db-49ed550f6fa3
begin
	res44 = normal_one_probility(log.(data4.x), data4.y)
	scatter(res44.y_pre, res44.e, xlabel="fitting value", ylabel="residual", label="do log")
end

# ╔═╡ e3cfa076-1943-11eb-35ac-3110da30870f
html"<br></br>"

# ╔═╡ 9d324cb2-1944-11eb-38b1-ff70b0c41486
md"""
## 5.5
Bottle makers have recoverded the average number of defects due to gravel per 10,000 bottles and the number of weeks since the last furnace was repaired.

1. fitting the linear model and Standard test for model suitability
2. suggeat a reasionable transformation to slove the mistake in 1, fitting the after transformation and model suitability
"""

# ╔═╡ c769f1e0-1946-11eb-397e-99b41d51b867
data5 = DataFrame(number=[13.0,16.1,14.5,17.8,22.0,27.4,16.8,34.2,65.6,49.2,66.2,81.2,87.4,114.5], weeks=[4,5,6,7,8,9,10,11,12,13,14,15,16,17])

# ╔═╡ 2215d92e-1947-11eb-1840-4d92bcb14185
scatter(data5.weeks, data5.number, leg=:none, xlabel="weeks", ylabel="number")

# ╔═╡ 14111252-19a2-11eb-16c2-fbb07b4d574f
ols5 = lm(@formula(number ~ weeks), data5)

# ╔═╡ 2e46c27a-19a2-11eb-31f6-cb94419d9a62
begin
	res5 = normal_one_probility(data5.weeks, data5.number)
	scatter(res5.y_pre, res5.e, xlabel="fitting value", ylabel="residual")
end

# ╔═╡ 54c592f0-19a2-11eb-0126-21695f895229
NormalProbabilityPlot(res5.e)

# ╔═╡ 87fe612e-19a2-11eb-1513-a3aafbf41113
begin
	res55 = normal_one_probility(data5.weeks, log.(data5.number))
	plot(scatter(res55.y_pre, res55.e, xlabel="fitting value", ylabel="residual", leg=:none), NormalProbabilityPlot(res55.e))
end

# ╔═╡ 46694f30-1b91-11eb-269f-6dede05ef8fb
scatter(data5.weeks, log.(data5.number), leg=:none, xlabel="weeks", ylabel="log(numbers)")

# ╔═╡ f87c19ba-1b92-11eb-295c-41e43149192c
begin
	change_data5 = DataFrame(weeks = data5.weeks, numbers = log.(data5.number))
	ols55 = lm(@formula(numbers ~ weeks), change_data5)
end

# ╔═╡ 47b98d3a-1b93-11eb-12c1-1d21ee1f2305
correct(md"""so the function: $ln(numbers) = 1.71622 + 0.1735\cdot weeks$""")

# ╔═╡ 839c68d6-1b93-11eb-3fef-0db10bfe7575
html"<br></br>"

# ╔═╡ d389d55e-1b93-11eb-0664-e1438da2e017
md"""
## 5.6
Consider the fuel consumpsion data in table B-18, ignoring the regreesion variable x₁, and recalling the complete residual analysis done in exercise 4.27, can transformation improve this analysis? why or why not?


> 考虑表B-18中的燃料消耗数据，忽略回归变量x1，回忆习题4.27所做的完整的残差分析，变换能改善这一分析吗？为什么能或为什么不能？
"""


# ╔═╡ 28e28770-1b95-11eb-0f93-b7709ed092d4
html"<br></br>"

# ╔═╡ 30098fc6-1b95-11eb-362e-bddf13d1e65b
md"""
## 5.7
Consider the methanol oxidation data in Table B-20 and make a throunth analysis of this data.Recalling the residual analysis of this data in problem 4.29, Can transformation improve this analysis? why or why not?

>考虑表B-20中的甲醇氧化数据，对这一数据做彻底的分析，回忆习题4.29对这一数据所做的残差分析。变换能改善这一分析吗？为什么能或为什么不能？

"""

# ╔═╡ b8825eaa-1b95-11eb-2667-3d5b149e2792
html"<br></br>"

# ╔═╡ c4edfb5e-1b95-11eb-06cc-a3deca511348
md"""
## 5.8

Consider three models

1. $y = \beta_0 + \beta_1 ( 1/x) + \epsilon$
2. $1/y = \beta_0 + \beta_1x + \epsilon$
3. $y = x/(\beta_0 - \beta_1 x) + \epsilon$

All three of these models are linearized by reciprocal transformation, sketch out what features appear in the scatter plot that will allow you to pick one of the three models.
> 这三个模型都通过倒数变换而线性化， 画出草图， 散点图出现什么特征会让你挑选三种模型中的一种。
"""

# ╔═╡ bd9932e6-1b96-11eb-35c7-957ea3fef253
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
# ╟─b8ce3ebc-18f0-11eb-3dd5-3f799465f5ec
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
# ╟─9f737ec4-17b4-11eb-1384-351e209c6cb8
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
# ╟─b5499d0a-1929-11eb-346c-2fc9cda07625
# ╟─a85eb29c-192b-11eb-0d05-c1087fb1ee63
# ╠═f5672d3a-192b-11eb-271d-3f82c74c1fce
# ╟─80c4ddf0-192c-11eb-02eb-6b8e5ed1ecde
# ╠═88253dec-192c-11eb-3fe4-d3c9f3eb9882
# ╠═f1750ab0-192d-11eb-23c4-f9f82e4343e7
# ╠═a94f5610-192c-11eb-387f-87f04e2d3774
# ╠═b29e4abc-192e-11eb-23ce-c52e1b6c41e0
# ╟─c0f3ad04-192f-11eb-166b-59f0ccd50ce1
# ╠═9ac087a0-1930-11eb-28e9-e7886e5816bc
# ╠═b367c318-1930-11eb-0d0b-53498dd6eab0
# ╠═aceb07f2-1933-11eb-122b-4185d4d50031
# ╟─00a65f8a-1941-11eb-0ece-7beb73a126d6
# ╟─a74d3a02-1941-11eb-2192-510011d1f284
# ╟─dee284c2-1941-11eb-0dab-ab42595e7d56
# ╟─0102859c-1943-11eb-332e-3f896f4114e5
# ╟─32ab63f2-1943-11eb-2ece-03627d4a4e09
# ╟─4c06b20c-1943-11eb-3191-21e1fb66dbc8
# ╟─69ff0ebc-1943-11eb-101b-43f5f6d7ce18
# ╟─9afdcdfa-1943-11eb-26e4-eb1268dfe551
# ╟─6e64f52e-1944-11eb-05db-49ed550f6fa3
# ╟─e3cfa076-1943-11eb-35ac-3110da30870f
# ╟─9d324cb2-1944-11eb-38b1-ff70b0c41486
# ╟─c769f1e0-1946-11eb-397e-99b41d51b867
# ╟─2215d92e-1947-11eb-1840-4d92bcb14185
# ╟─14111252-19a2-11eb-16c2-fbb07b4d574f
# ╟─2e46c27a-19a2-11eb-31f6-cb94419d9a62
# ╟─54c592f0-19a2-11eb-0126-21695f895229
# ╟─87fe612e-19a2-11eb-1513-a3aafbf41113
# ╟─46694f30-1b91-11eb-269f-6dede05ef8fb
# ╟─f87c19ba-1b92-11eb-295c-41e43149192c
# ╟─47b98d3a-1b93-11eb-12c1-1d21ee1f2305
# ╟─839c68d6-1b93-11eb-3fef-0db10bfe7575
# ╟─d389d55e-1b93-11eb-0664-e1438da2e017
# ╟─28e28770-1b95-11eb-0f93-b7709ed092d4
# ╟─30098fc6-1b95-11eb-362e-bddf13d1e65b
# ╟─b8825eaa-1b95-11eb-2667-3d5b149e2792
# ╟─c4edfb5e-1b95-11eb-06cc-a3deca511348
# ╟─bd9932e6-1b96-11eb-35c7-957ea3fef253
