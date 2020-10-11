### A Pluto.jl notebook ###
# v0.12.1

using Markdown
using InteractiveUtils

# ╔═╡ bd3bbd0c-056e-11eb-05c3-1dae71fbcfd3
using DataFrames

# ╔═╡ f5b05dfa-056e-11eb-3e75-49e213daff01
using CSV

# ╔═╡ fc1efae4-0577-11eb-17df-ddcf53b193cb
using DelimitedFiles

# ╔═╡ 0a043ca0-0578-11eb-151b-2d706034d2cc
using BenchmarkTools

# ╔═╡ 98ec3f42-0581-11eb-112d-973e42d69028
using Queryverse

# ╔═╡ 7e79203c-0591-11eb-183b-3d463db683a1
using Dates

# ╔═╡ 0890fdf0-0593-11eb-2617-8993a9014069
using Plots

# ╔═╡ 7e184ee0-084a-11eb-26d1-3f27e600df7d
using Flux

# ╔═╡ 9834bb80-084a-11eb-37de-412fbca3cee7
using Flux: @epochs

# ╔═╡ a375f890-084a-11eb-1ec9-ddea0b709d31
using IterTools: ncycle

# ╔═╡ b0ff871a-084a-11eb-11f3-55f601c5f63f
using Parameters: @with_kw

# ╔═╡ c02c25c2-084a-11eb-15a6-cda4fb28c236
using WebIO

# ╔═╡ 950cbb44-0562-11eb-1949-e3d9165a2436
md" # Covid-19, Analysis, Visualization, Prediction"

# ╔═╡ 42e72a4c-0563-11eb-10f8-b5db40ef30a8
md"**Author: Zhuofei, Zhou**

***"

# ╔═╡ 774be886-0563-11eb-0e13-31a4fe1d9271
md"## Introduction
### Covid-19



![Covid-19](https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1601719355&di=00632f6975b3b3d60acdef92bcd8fd25&src=http://www.yxppt.com/wp-content/uploads/2020/02/24/334627/yxppt-2020-02-24_12-48-48_812888.jpg)

Novel Coronavirus 2019, on 12 January 2020, WHO officially named it 2019-NCOV. Coronaviruses are a large family of viruses known to cause colds and more serious illnesses such as Middle East Respiratory syndrome (MERS) and severe acute respiratory syndrome (SARS). Novel Coronavirus is a novel coronavirus strain that has never been found in humans before. More information is available.[百度百科](https://baike.baidu.com/item/2019新型冠状病毒/24267858?fr=aladdin)

### data

Data from COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.[link](https://github.com/CSSEGISandData/COVID-19)
- time series covid19 confirmed global.csv.([github link](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv)).
- time series covid19 deaths global.csv.([github link](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv))
- time serise covid19 recoverd global.csv([github link](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv))

- cases contry.csv.([github link](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv))

- cases time.csv. ([github link](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv))


download here:

"

# ╔═╡ 27e37c60-0582-11eb-1d76-05359f6465a4
md"
***
import some packages in julia:"

# ╔═╡ 885c8cee-0602-11eb-078c-bf0c90d22d00
pgfplotsx()

# ╔═╡ 5056ae0c-084a-11eb-2940-297a3406887e
import Plots.plot

# ╔═╡ 5a93b1bc-084a-11eb-1e37-d3665d2c30a8
import PlotlyJS

# ╔═╡ ef325640-059b-11eb-02e1-85c67610f89b
md"***"

# ╔═╡ 817a46da-056e-11eb-0d48-97ee88055ce0
comfirmed = load("../datasets/time_series_covid19_comfirmed_global.csv") |> DataFrame

# ╔═╡ 6ad7e4be-058c-11eb-3fac-35f5e9f14513
dt, Header = readdlm("../datasets/time_series_covid19_comfirmed_global.csv", ',', header=true)

# ╔═╡ f88187b8-0594-11eb-3181-41bdee367c4a
dt_deaths, Header_deaths = readdlm("../datasets/time_series_covid19_deaths_global.csv", ',', header=true)

# ╔═╡ 1be24c26-05fc-11eb-15e8-81d8904a38ab
dt_recover,Header_recov = readdlm("../datasets/time_series_covid19_recovered_global.csv",',',header=true)

# ╔═╡ 15ed28e6-059c-11eb-190c-2512c8cc9d8e
md"## Analysis
### Analysis Global Data
Now, we just use the data to rebuild a new dataframe to store the Global  data.
I create a new dataframe named as **Global_num** to store the number of deaths and comfirmed numbers，time from 2020-01-22 to 2020-10-02.

#### Data Operate
operating as folows:"

# ╔═╡ 1037f522-0590-11eb-2a1e-318b94418794
begin
	confirmed_num = []
	deaths_num = []
	recovered_num = []
	for i in 5:259
		push!(confirmed_num, sum(dt[:, i]))
		push!(deaths_num, sum(dt_deaths[:, i]))
		push!(recovered_num, sum(dt_recover[:,i]))
	end
	dates = Date(2020, 1, 22):Day(1):Date(2020, 10, 2)
end

# ╔═╡ b8eb5ea0-0592-11eb-2240-ff69390ac636
Global_num = DataFrame(Date=dates, confirmed_num=confirmed_num, deaths_num=deaths_num, recovered_num=recovered_num)

# ╔═╡ a1753130-0601-11eb-1371-ab5cf2290e5b
begin
	n, = size(confirmed_num)
	new_confirmed = []
	new_death = []
	new_recovered = []
	death_rate = deaths_num ./ confirmed_num
	recover_rate = recovered_num ./ confirmed_num
	for i in 2:n
		push!(new_confirmed, confirmed_num[i]-confirmed_num[i-1])
		push!(new_death, deaths_num[i] - deaths_num[i-1])
		push!(new_recovered, recovered_num[i]-recovered_num[i-1])
	end
	Global_num.death_rate = death_rate
	Global_num.recover_rate = recover_rate
end

# ╔═╡ 40d40f2e-059d-11eb-05a2-47b885c74f74
md"
#### Plot show
Then i plot the number of comfirmed cases and deaths cases over time.

As we can see, the number of comfirmed cases continues to increase exponentially. But the number of deaths cases seems to increase linearly.
"

# ╔═╡ 7cb20d98-0593-11eb-1ad9-37117ba3e5a0
p1 = plot(dates, confirmed_num, legend=:none, ylabel="Numbers", xlabel="Date", title="Global comfirmed numbers")

# ╔═╡ 56f43390-0595-11eb-116d-e10adb862fea
p2 = plot(dates, deaths_num, legend=:none, linecolor=:red, ylabel="Numbers", xlabel="Date", title="Global Deaths numbers")

# ╔═╡ 990e195a-05fc-11eb-37c9-d15175e25493
p_re = plot(dates, recovered_num, legend=:none, linecolor=:green, ylabel="Numbers", xlabel="Date", title="Global recovered numbers")

# ╔═╡ 6b000e4a-0595-11eb-1d6c-6fb54a74b93f
begin
	p3 = plot(dates, [confirmed_num deaths_num recovered_num], label=["Confirmed" "Deaths" "recovered"], legend=:topleft)
	l = @layout([a [b;c;d]])
	plot(p3, p1, p2,p_re, layout=l,xticks = [Date(2020, 1, 22), Date(2020,10,2)],size=(700,400), ylabel="Numbers", xlabel="Date", titlefontsize=10, xaxis=(font(8)))
end

# ╔═╡ f7e7f280-05f1-11eb-1d21-71674a8adbc0
md"Now, start calculate the New cases every day，and the deaths rate."

# ╔═╡ 8f2f1d20-05f3-11eb-25b0-1784220d2499
p4 = plot(Date(2020, 1, 23):Day(1):Date(2020, 10, 2),new_confirmed,shape=:circle, legend=:topleft, label="new case", xlabel="Date", ylabel="Numbers", title="new cases every day")

# ╔═╡ c6d41bda-05f9-11eb-347b-6f367c5b70fb
p5 = plot(Date(2020,1,23):Day(1):Date(2020,10,2), new_death,linecolor=:red,shape=:circle, legend=:topleft, label="new death", xlabel="Date", ylabel="Numbers", title="New death every day")

# ╔═╡ 116a25a6-05fd-11eb-2969-c37e38fab7ae
p6 = plot(Date(2020,1,23):Day(1):Date(2020,10,2), new_recovered,shape=:circle,linecolor=:green, legend=:topleft, label="new recovered", xlabel="Date", ylabel="Numbers", title="New recoverd every day")

# ╔═╡ 0e76b13c-05fa-11eb-3ec2-d76b973dbf5a
Global_num[Global_num.death_rate .== maximum(Global_num.death_rate), :]

# ╔═╡ 1ac7c3b8-05f5-11eb-048c-47188f25dd78
begin
	plot(Date(2020, 1, 22):Day(1):Date(2020, 10, 2),death_rate, linecolor=:red, title="Death rate", xlabel="date", ylabel="Numbers", legend=:none, ylims=(0,0.08))
	plot!([Date(2020,1,22),Date(2020,4,29),Date(2020,4,29)], [maximum(Global_num.death_rate),maximum(Global_num.death_rate), 0], 
    line=:dot, linecolor=:black, annotations=[(Date(2020,4,29), 0.075, text("max death rate: 0.0723", 10))])
end

# ╔═╡ ba3e86c2-05fd-11eb-1431-354e2a567533
plot(Date(2020, 1, 22):Day(1):Date(2020, 10, 2),[death_rate recover_rate],xlabel="date", ylabel="Numbers", legend=:outerright, linecolor=[:red :green], title="death rate and recover rate", label=["death" "recover"])

# ╔═╡ b0ece2e4-05f8-11eb-30e4-39e726ac635c
md"the maximum death rate in 2020-4-29

So, the new comfirmed cases continues to rise, the death rate is already falling."

# ╔═╡ a41cc41c-08a5-11eb-04ed-43bbcf5b8bc8
md"
#### Prediction
**Global Trend:**

It is useful to understand the global trend of an increase in the number of cases over time. There is always a pattern in any data, but the concern is how strongly data follows a pattern. COVID-19 spreads exponentially.

i think of some ways to estimate the curve:

- Numerical Analysis: ❎
- linear regression, after transmission: ✅
- neural network: ✅

So, we first focus on comfirmed case:

do $y'=\sqrt{y}$ when i use to transition. maybe will get a linea function.:
"

# ╔═╡ 074a5f34-08a8-11eb-2e6f-3fe69c3be774
begin
	#transition
	x = 0:n-1
	y = Global_num.confirmed_num
	y₁ = sqrt.(y)
	X = [x ones(n)]
	β = inv(X'*X)*X'*y₁
	ŷ = (X*β).^2
	plot(dates,Global_num.confirmed_num, label="real")
	plot!(dates, ŷ, label="predict", annotations=[(Date(2020,4,22), 30000000, text("predict function: y=(24.77t - 584.29)²", 10))])
end

# ╔═╡ 690f7ea2-08a8-11eb-3a77-c32e93b0b950
md"""
Other ways: _**Neural Network**_

`import Flux` to do meachine learning.

the all code here:
```julia
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: throttle
using Parameters: @with_kw
using DelimitedFiles
using IterTools: ncycle
using Dates
using DataFrames
using CSV
using Plots


@with_kw mutable struct Args
    η::Float64 = 0.001
    batchsize::Int = 1
    epochs::Int = 1000
end
cd(@__DIR__)
pwd()
#read data:
dt, Header = readdlm("time_series_covid19_comfirmed_global.csv", ',', header=true)
dt_deaths, Header_deaths = readdlm("time_series_covid19_deaths_global.csv", ',', header=true)
dt_recover,Header_recov = readdlm("time_series_covid19_recovered_global.csv",',',header=true)


#create a new datafram to store datum
comfirmed_num = []
deaths_num = []
recovered_num = []
for i in 5:259
    push!(comfirmed_num, sum(dt[:, i]))
    push!(deaths_num, sum(dt_deaths[:, i]))
    push!(recovered_num, sum(dt_recover[:,i]))
end
dates = Date(2020, 1, 22):Day(1):Date(2020, 10, 2)
Global_num = DataFrame(Date=dates, 
    comfirmed_num=comfirmed_num, 
    deaths_num=deaths_num, 
    recovered_num=recovered_num)

#read train_data
n, = size(comfirmed_num)
x = 0:n-1
y = Global_num.comfirmed_num
args = Args()
train_data = DataLoader((Array(x), Float64.(y)), batchsize=args.batchsize)

# define leaky relu
Lelu(x, α=100) = (x ≥ 0 ? x : x/α)

#define Model
m = Chain(
    Dense(1, 40, Lelu),
    Dense(40, 40, Lelu),
    Dense(40, 40, Lelu),
    Dense(40, 1, Lelu)
)

#define loss function
loss(x, y) = Flux.mse(m(x), y)

#define parameters
ps = Flux.params(m)

#define Opt
opt = ADAM(args.η)

#train model
Flux.train!(loss, ps, ncycle(train_data, args.epochs), opt)

#visualize
flux_y = []
for i in Array(x)
    push!(flux_y, Array(m([i]))[1])
end
flux_y = Float64.(flux_y)
plot(dates, [flux_y Float64.(y)], 
    label=["predict" "real"], xlabel="date", ylabel="Numbers", size=(900, 600))
```

"""

# ╔═╡ 87b583e0-08aa-11eb-0cd9-396999de8d5b
#write a function
function predict_curve(x, y, η::Float64=0.001, epochs::Int=1000, batchsize::Int=1)
    dates = Date(2020, 1, 22):Day(1):Date(2020, 10, 2)
	#prepare data
    train_data = Flux.Data.DataLoader((x, y), batchsize=batchsize)
	#define leaky relu
    Lelu(x, α=100) = (x ≥ 0 ? x : x/α)
	
	#define model
    m = Chain(
    Dense(1, 40, Lelu),
    Dense(40, 40, Lelu),
    Dense(40, 40, Lelu),
    Dense(40, 1, Lelu))
	
	#define mse
    loss(x, y) = Flux.mse(m(x), y)
	
	#parameters
    ps = Flux.params(m)
	
    opt = ADAM(η)
    Flux.train!(loss, ps, ncycle(train_data, epochs), opt)
    flux_y = []
    for i in Array(x)
        push!(flux_y, Array(m([i]))[1])
    end
    plot(dates, [Float64.(flux_y) Float64.(y)],label=["predict" "real"], xlabel="date", ylabel="Numbers", legend=:topleft)
end

# ╔═╡ 18d4a4fa-08ab-11eb-2912-79d0c89add25
predict_curve(Array(0:254), Float64.(Global_num.confirmed_num))

# ╔═╡ a1cc6374-08ab-11eb-2b6e-e1e9d6848603
predict_curve(Array(0:254), Float64.(Global_num.deaths_num))

# ╔═╡ f499186a-08ab-11eb-271d-277d68f0c0da
predict_curve(Array(0:254), Float64.(Global_num.recovered_num))

# ╔═╡ 30d23aa2-08ad-11eb-174d-bdaab2b5bb8d
md"
### Analysis country data

consider analyzing the situation in different countries.
"

# ╔═╡ ccaa2948-08b3-11eb-1848-c9006ef99374
dt_country = CSV.read("../datasets/cases_country.csv")

# ╔═╡ 597c6202-08ad-11eb-241f-75f2794fe332
sort(dt_country, [:Confirmed], rev=true)

# ╔═╡ a4862d54-08b3-11eb-1887-ab22f0db7600
dt_country

# ╔═╡ Cell order:
# ╟─950cbb44-0562-11eb-1949-e3d9165a2436
# ╟─42e72a4c-0563-11eb-10f8-b5db40ef30a8
# ╟─774be886-0563-11eb-0e13-31a4fe1d9271
# ╟─27e37c60-0582-11eb-1d76-05359f6465a4
# ╠═bd3bbd0c-056e-11eb-05c3-1dae71fbcfd3
# ╠═f5b05dfa-056e-11eb-3e75-49e213daff01
# ╠═fc1efae4-0577-11eb-17df-ddcf53b193cb
# ╠═0a043ca0-0578-11eb-151b-2d706034d2cc
# ╠═98ec3f42-0581-11eb-112d-973e42d69028
# ╠═7e79203c-0591-11eb-183b-3d463db683a1
# ╠═0890fdf0-0593-11eb-2617-8993a9014069
# ╠═885c8cee-0602-11eb-078c-bf0c90d22d00
# ╠═5056ae0c-084a-11eb-2940-297a3406887e
# ╠═5a93b1bc-084a-11eb-1e37-d3665d2c30a8
# ╠═7e184ee0-084a-11eb-26d1-3f27e600df7d
# ╠═9834bb80-084a-11eb-37de-412fbca3cee7
# ╠═a375f890-084a-11eb-1ec9-ddea0b709d31
# ╠═b0ff871a-084a-11eb-11f3-55f601c5f63f
# ╠═c02c25c2-084a-11eb-15a6-cda4fb28c236
# ╟─ef325640-059b-11eb-02e1-85c67610f89b
# ╠═817a46da-056e-11eb-0d48-97ee88055ce0
# ╠═6ad7e4be-058c-11eb-3fac-35f5e9f14513
# ╠═f88187b8-0594-11eb-3181-41bdee367c4a
# ╠═1be24c26-05fc-11eb-15e8-81d8904a38ab
# ╟─15ed28e6-059c-11eb-190c-2512c8cc9d8e
# ╠═1037f522-0590-11eb-2a1e-318b94418794
# ╠═b8eb5ea0-0592-11eb-2240-ff69390ac636
# ╠═a1753130-0601-11eb-1371-ab5cf2290e5b
# ╟─40d40f2e-059d-11eb-05a2-47b885c74f74
# ╠═7cb20d98-0593-11eb-1ad9-37117ba3e5a0
# ╠═56f43390-0595-11eb-116d-e10adb862fea
# ╟─990e195a-05fc-11eb-37c9-d15175e25493
# ╠═6b000e4a-0595-11eb-1d6c-6fb54a74b93f
# ╟─f7e7f280-05f1-11eb-1d21-71674a8adbc0
# ╠═8f2f1d20-05f3-11eb-25b0-1784220d2499
# ╠═c6d41bda-05f9-11eb-347b-6f367c5b70fb
# ╠═116a25a6-05fd-11eb-2969-c37e38fab7ae
# ╠═0e76b13c-05fa-11eb-3ec2-d76b973dbf5a
# ╠═1ac7c3b8-05f5-11eb-048c-47188f25dd78
# ╟─ba3e86c2-05fd-11eb-1431-354e2a567533
# ╟─b0ece2e4-05f8-11eb-30e4-39e726ac635c
# ╟─a41cc41c-08a5-11eb-04ed-43bbcf5b8bc8
# ╠═074a5f34-08a8-11eb-2e6f-3fe69c3be774
# ╟─690f7ea2-08a8-11eb-3a77-c32e93b0b950
# ╟─87b583e0-08aa-11eb-0cd9-396999de8d5b
# ╠═18d4a4fa-08ab-11eb-2912-79d0c89add25
# ╠═a1cc6374-08ab-11eb-2b6e-e1e9d6848603
# ╠═f499186a-08ab-11eb-271d-277d68f0c0da
# ╟─30d23aa2-08ad-11eb-174d-bdaab2b5bb8d
# ╠═ccaa2948-08b3-11eb-1848-c9006ef99374
# ╠═597c6202-08ad-11eb-241f-75f2794fe332
# ╠═a4862d54-08b3-11eb-1887-ab22f0db7600
