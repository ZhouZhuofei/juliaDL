### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ d627801e-3479-11eb-1fab-1dfaf4af1144
begin
	using Random
	Random.seed!()
	
	passLength, numMatchesForLog = 8, 4
	possibleChars = ['a':'z' ; 'A':'Z' ; '0':'9']
	
	correctPassword = "3xsfe6p3"
	
	numMatch(loginPassword) = sum([loginPassword[i] == correctPassword[i] for i in 1:passLength])
	
	Nₚ = 10^7
	
	passwords = [String(rand(possibleChars, passLength)) for _ in 1:Nₚ]
	numLogs = sum([numMatch(p) >= numMatchesForLog for p in passwords])
end

# ╔═╡ 81248f92-347b-11eb-046d-1327fefd71be
begin
	using StatsBase, Combinatorics, Plots; gr()
	matchExists1(n) = 1 - prod([k/365 for k in 365:-1:365-n+1])
	matchExists2(n) = 1 - factorial(365, 365 - big(n))/365^big(n)
	function bdEvent(n)
		birthdays = rand(1:365, n)
		dayCounts = counts(birthdays, 1:365)
		return maximum(dayCounts) > 1
	end
	
	probEst(n) = sum([bdEvent(n) for _ in 1:Nᵦ]) / Nᵦ
	
	xGrid = 1:50
	analyticSolution1 = [matchExists1(n) for n in xGrid]
	analyticSolution2 = [matchExists2(n) for n in xGrid]
	
	Nᵦ = 10^3
	mcEstimates = [probEst(n) for n in xGrid]
	plot(xGrid, analyticSolution1, c=:blue, label="Analytic solution")
	scatter!(xGrid, mcEstimates, c=:red, ms=4, msw=1, shape=:xcross, label="MC estimate", xlims=(0, 50), ylim=(0, 1), xlabel="Number of people in room", ylabel="Probaility of birthday match", leg=:topleft)
	
		
end

# ╔═╡ fcc139c4-3481-11eb-3765-6fe44d8b9951
begin
	using LaTeXStrings
	
	
	nₗ, Nₗ = 5, 10^5
	
	function isUpperLattice(v)
		for i in 1:Int(length(v)/2)
			sum(v[1:2*i-1]) >= i ? continue : return false
		end
		return true
	end
	
	omega = unique(permutations([zeros(Int, nₗ);ones(Int, nₗ)]))
	A = omega[isUpperLattice.(omega)]
	pA_modelI = length(A)/length(omega)
	
	function randomWalkPath(n)
		x, y = 0, 0
		path = []
		while x<n && y<n
			if rand() < 0.5
				x += 1
				push!(path, 0)
			else
				y += 1
				push!(path, 1)
			end
		end
		append!(path, x<n ? zeros(Int64, n-x) : ones(Int64, n-y))
		return path
	end
	
	PA_modelIIest = sum([isUpperLattice(randomWalkPath(nₗ)) for _ in 1:Nₗ]) / Nₗ
	
	function plotPath(v, l, c)
		x, y = 0, 0
		graphX, graphY = [x], [y]
		for i in v
			if i == 0
				x += 1
			else
				y += 1
			end
			push!(graphX, x), push!(graphY, y)
		end
		plot!(graphX, graphY,
			la=0.8, lw = 2, label=l, c=c, ratio = :equal, legend=:topleft,xlims=(0,nₗ), ylims=(0,nₗ), xlabel="East rightarrow", ylabel="North rightarrow")
	end
	plot()
	plotPath(rand(A), "Upper lattice path", :blue)
	plotPath(rand(setdiff(omega, A)), "Non-Upper lattice path", :red)
	plot!([0, nₗ], [0,nₗ],ls=:dash,c=:black, label="")
	
end
	
	

# ╔═╡ 2462b3a2-3478-11eb-2047-d17d9419ef0a
md"# _Probaility_"

# ╔═╡ 5235dc84-3478-11eb-2f71-11b5b2f75b44
md"
1. A sample space
2. A collection of events
3. A probaility measure
"

# ╔═╡ 8e828a70-3478-11eb-2539-f5be394fbbdf
md"## Event sum of dice"

# ╔═╡ a8ab5e90-3478-11eb-3f35-339109ff8c54
begin
	Nₜ, faces = 10^6, 1:6
	numSol = sum([iseven(i+j) for i in faces, j in faces]) / length(faces)^2
	mcest = sum([iseven(rand(faces) + rand(faces)) for i in 1:Nₜ]) / Nₜ
	println("Numerical solution = $numSol \n Monte Carlo estimate = $mcest")
end

# ╔═╡ 7c6ca8f6-3479-11eb-3f7c-d35794372677
numSol

# ╔═╡ 826c806e-3479-11eb-022d-53c8acda5a74
mcest

# ╔═╡ c95a38d6-3479-11eb-2aed-d9ce01229a1f
md"## Password matching"

# ╔═╡ 5b1c6338-347b-11eb-043a-49549cd35c32
md"## The Birthday Problem"

# ╔═╡ 592036ba-347e-11eb-07da-1134d98b2a0f
md"## Fishing with and without replacement"

# ╔═╡ 5f033766-347f-11eb-3dce-9d840202929b
begin
	function proportionFished(gF, sF, n, N, withReplacement = false)
		function fishing()
			fishInPond = [ones(Int64, gF); zeros(Int64, sF)]
			fishCaught = Int64[]
			
			for fish in 1:n
				fished = rand(fishInPond)
				push!(fishCaught, fished)
				if withReplacement == false
					deleteat!(fishInPond, findfirst(x->x==fished, fishInPond))
				end
			end
			sum(fishCaught)
		end
		simulations = [fishing() for _ in 1:N]
		proportions = counts(simulations, 0:n)/N
		
		if withReplacement
			plot!(0:n, proportions,line=:stem, marker=:circle, c=:blue, ms=4, msw=1, label="With Replacement", xlabel="n", ylabel="Probaility", ylims=(0, 0.6))
		else
			plot!(0:n, proportions, line=:stem, marker=:xcross, c=:red, msw=4, label="Without replacement")
		end
	end
	
	N₄ = 10^6
	goldFish, silverFish, n = 3, 4, 3
	plot()
	proportionFished(goldFish, silverFish, n, N₄)
	proportionFished(goldFish, silverFish, n, N₄, true)
end
		

# ╔═╡ ac4a60ee-347f-11eb-2580-659176e69bb2
md"## Lattice paths"

# ╔═╡ abacb3da-347f-11eb-06fd-df1a2f6812b6


# ╔═╡ ab179624-347f-11eb-141e-815098dddcd0


# ╔═╡ Cell order:
# ╟─2462b3a2-3478-11eb-2047-d17d9419ef0a
# ╟─5235dc84-3478-11eb-2f71-11b5b2f75b44
# ╟─8e828a70-3478-11eb-2539-f5be394fbbdf
# ╠═a8ab5e90-3478-11eb-3f35-339109ff8c54
# ╠═7c6ca8f6-3479-11eb-3f7c-d35794372677
# ╠═826c806e-3479-11eb-022d-53c8acda5a74
# ╟─c95a38d6-3479-11eb-2aed-d9ce01229a1f
# ╠═d627801e-3479-11eb-1fab-1dfaf4af1144
# ╟─5b1c6338-347b-11eb-043a-49549cd35c32
# ╠═81248f92-347b-11eb-046d-1327fefd71be
# ╟─592036ba-347e-11eb-07da-1134d98b2a0f
# ╠═5f033766-347f-11eb-3dce-9d840202929b
# ╟─ac4a60ee-347f-11eb-2580-659176e69bb2
# ╠═fcc139c4-3481-11eb-3765-6fe44d8b9951
# ╠═abacb3da-347f-11eb-06fd-df1a2f6812b6
# ╠═ab179624-347f-11eb-141e-815098dddcd0
