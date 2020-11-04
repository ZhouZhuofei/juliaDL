### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ eceac3a0-1e4a-11eb-19d7-e1a6af47842e
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["DifferentialEquations", "Plots", "PlutoUI", "Distributions"])
	using DifferentialEquations
	using PlutoUI
	using Plots
	using Distributions
end

# ╔═╡ f880960c-1eb5-11eb-08d4-656728e0297e
md"_*reading notes from 《METABOLIC GROWTH THEORY》, book author: Chen ping.*_"

# ╔═╡ 506c8354-1e42-11eb-3537-d9c055da2552
md"## Problems?"

# ╔═╡ e1ee8568-1e40-11eb-00d2-3d279f8e9bb3
md"_**What is science?**_"

# ╔═╡ d1924e1a-1e41-11eb-3640-ab6e3784c8ec
md"> Science is an institution, it is a method, it is an accumulated tradition of kownledge, it is a major factor in the development of production, and it is a powerful force in the information of the universe"

# ╔═╡ 59efff32-1e42-11eb-224f-bd5a0929bd69
md"_**China led the world in science, but lagged behind only after the Ming and Qing Dynasties?**_"

# ╔═╡ a8ca358c-1e42-11eb-0122-4dedb09ca5e1
md"> so we say behind only after the Ming and Qing Dynasties, we are behind the Western. it will reduce the stigma, but it impedes our understanding of the weaknesses of traditional chinese secience.
> China was always backward in the scientific systerm and the foundation of scientific method, and the independent and complete scientific system never emerged from the feudal society.Not to admit the backwardness of system and method was just the staring point of the backwardness of modern china."

# ╔═╡ e13f067a-1e44-11eb-044f-e5dc23a70983
md"_**What is Technology?**_"

# ╔═╡ 04f3d500-1e45-11eb-2ed6-1b1057a090b8
md"> Technology is the use of scientific methods and knowledge to solve the problems facing human life in order to obtain the greatest possible economic benefits."

# ╔═╡ 704a389a-1e46-11eb-2717-f17fc35381e0
md"# Content"

# ╔═╡ 867332b4-1e46-11eb-1e49-470a9cdded52
md"## The Origin of division of Labor and a stochastic model of social differentiation"

# ╔═╡ 456b046c-1e47-11eb-0aea-510bda71e16e
md"""
We note that **_cultural factors_** played an important role in the origins of capitalism and modern science.

he proposed a one-dimensional model of individualism, assuming a horizontal axis, with highly individualistic Europe and the United States at one extreme and societies with rigid division of Labour like bees at the other.(Kikuchi, 1981)

**_Information diffusion model_**:Use Logistic equation to sociology.

$\frac{\partial n}{\partial t} = kn(N-n) - dn(1 - \alpha \frac{n}{N})$

noted:$N$ represents the population size, $n$ is the number of people who have mastered the information, $(N-n)$ is the number of learners who need to receive new information, and $k$ is the growth rate of the number of people who receive new information in the learning process. The Last term represents the elimination rate or forgetting rate. $\alpha$ is a measure of how repulsive people are to unfamiliar things
"""

# ╔═╡ 522f7894-1e73-11eb-0131-65336d6da211
md"""

so $N$ : $(@bind N Slider(10:20; default=10, show_value=true))

const $k$ : $(@bind k Slider(0.00 : 0.01 : 0.50; default=0.02, show_value=true))

rate α : $(@bind α Slider(-0.50 : 0.01 : 0.50; default=0.00, show_value=true))
"""

# ╔═╡ 39d1607e-1e74-11eb-2c9c-ab49addfdc1d
begin
	Infordiff(u,p,t)=k*u*(N-u) - 0.02 * u*(1.0 - α* u/N)
	u0 = 0.1
	tspan = (0.0,100.0)
	prob = ODEProblem(Infordiff,u0,tspan)
	sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
	plot(sol)
end

# ╔═╡ b6bce1e4-1e74-11eb-2ba5-8f641457daa8
function Infor(ina)
	plot(location=4,legend = :outertopright)
	for i in ina
		Infordiff1(u,p,t)=0.02*u*(10-u) - 0.1u * (1 - i * u/10)
		u0 = 0.1
		tspan = (0.0,100.0)
		prob = ODEProblem(Infordiff1,u0,tspan)
		sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
		plot!(sol, label="α = $(i)")
	end
	plot!(axis=false)
end
	
	

# ╔═╡ 421f5f96-1e75-11eb-0783-251da9b8ca47
Infor(-0.9:0.2:0.9)

# ╔═╡ 0f29b070-1eb4-11eb-28bb-d723b952495e
md"""we can solve that if the $-1<\alpha <0$, $\alpha$ is a measure of how repulsive people are to unfamiliar things. use it to measure society's preference for risky behavior, more closely to $-1$, you are more like risky or you can more receptive to new thing.

if $\alpha$ in $[0,1]$,we measure pepole's rejection of unfamiliar things.

from the plot above, we can see the number of equilibrium from $-0.9$ to $0.9$,the $n$ is different,the smaller $\alpha$ will get smaller equilibrium $n$.

**_Conclusion_: $n_{\alpha <0} < n_{\alpha=0} < n_{\alpha>0}$, The balenced population or market size of the individualist culture is smaller than that of the collectivist culture.**
"""

# ╔═╡ fb3c4b76-1eb4-11eb-2824-4104209213b2
md" ## A fluctuating environment"

# ╔═╡ a2ac3c50-1eb6-11eb-17cd-359e9d98b756
md"if you are dealing with a fluctuating environment, use this function, here:"

# ╔═╡ Cell order:
# ╟─f880960c-1eb5-11eb-08d4-656728e0297e
# ╟─506c8354-1e42-11eb-3537-d9c055da2552
# ╟─e1ee8568-1e40-11eb-00d2-3d279f8e9bb3
# ╟─d1924e1a-1e41-11eb-3640-ab6e3784c8ec
# ╟─59efff32-1e42-11eb-224f-bd5a0929bd69
# ╟─a8ca358c-1e42-11eb-0122-4dedb09ca5e1
# ╟─e13f067a-1e44-11eb-044f-e5dc23a70983
# ╟─04f3d500-1e45-11eb-2ed6-1b1057a090b8
# ╟─704a389a-1e46-11eb-2717-f17fc35381e0
# ╟─867332b4-1e46-11eb-1e49-470a9cdded52
# ╟─456b046c-1e47-11eb-0aea-510bda71e16e
# ╟─eceac3a0-1e4a-11eb-19d7-e1a6af47842e
# ╟─522f7894-1e73-11eb-0131-65336d6da211
# ╟─39d1607e-1e74-11eb-2c9c-ab49addfdc1d
# ╟─b6bce1e4-1e74-11eb-2ba5-8f641457daa8
# ╟─421f5f96-1e75-11eb-0783-251da9b8ca47
# ╟─0f29b070-1eb4-11eb-28bb-d723b952495e
# ╟─fb3c4b76-1eb4-11eb-2824-4104209213b2
# ╟─a2ac3c50-1eb6-11eb-17cd-359e9d98b756
