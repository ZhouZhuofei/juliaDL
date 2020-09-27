### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 71c5500c-00cc-11eb-3162-a10318958fc2
using Flux

# ╔═╡ 703d8bca-00cd-11eb-1b57-af64a482bd8a
using LinearAlgebra

# ╔═╡ 78ac553e-00cd-11eb-26b7-f5bdbe229dd7
using Gadfly

# ╔═╡ 9ae94d2a-00cd-11eb-37b9-1dc46a16a49d
using Statistics

# ╔═╡ a23e157c-00cd-11eb-268c-d992c466fdd8
using Distributions

# ╔═╡ bd0aafe6-00cd-11eb-21ca-fd6c71a68f2f
using Random

# ╔═╡ 34ebec92-00d2-11eb-3f06-c30d260fddda
using DataFrames

# ╔═╡ 0eb7608c-00cc-11eb-0abe-bd2bca0d2b3e
md"# Sigmod regression

use to clssification

The loss function : cross entropy function
"

# ╔═╡ afdc663e-00cd-11eb-04b7-899986c22aee
Random.seed!(123)

# ╔═╡ c37e9a4a-00cd-11eb-32ce-b514071f2623
GroupOne = rand(MvNormal([10.0, 10.0], 10.0 * Matrix{Float64}(I, 2, 2)), 100)

# ╔═╡ cc6f4438-00cd-11eb-01c8-578b14b406b6
GroupTwo = rand(MvNormal([0.0, 0.0], 10 * Matrix{Float64}(I, 2, 2)), 100)

# ╔═╡ df38d270-00cd-11eb-2899-419fbf880361
X = Flux.normalise(hcat(GroupOne, GroupTwo), dims=2)

# ╔═╡ 2a87f08e-00d2-11eb-2889-1f2385032787
dt = DataFrame()

# ╔═╡ 44233b5c-00d2-11eb-3bc6-47248934072c
dt.x = X[1, :]

# ╔═╡ 50de7302-00d2-11eb-183a-752f712cbb38
dt.y = X[2, :]

# ╔═╡ 7177531a-00d2-11eb-1f6c-53d12ebffb5c
plot(dt, x=:x, y=:y, color=:label)

# ╔═╡ ffd9738e-00cd-11eb-0681-4d71a42001b1
begin
	y = []
	for i in 1:200
		if 1<=i<=100
			push!(y, "A")
		else
			push!(y, "B")
		end
	end
end

# ╔═╡ 57182a38-00d2-11eb-1aa3-f39349334c9c
dt.label = y

# ╔═╡ 258db6e4-00ce-11eb-08c7-0ff693c39c95
labels = Flux.onehotbatch(y, [:"A", :"B"])

# ╔═╡ 656fb6e0-00ce-11eb-014b-6d34bd579e77
train_data = Flux.Data.DataLoader((X, labels), batchsize=10, shuffle=true)

# ╔═╡ 808a8b6c-00ce-11eb-27f7-fb9c543b53ff
accuracy(x, y, model) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# ╔═╡ a24cb676-00ce-11eb-36cd-0f7f0e955ec1
m = Chain(Dense(2,2))

# ╔═╡ a7bb0414-00ce-11eb-35b6-e596f46224d3
loss(x,y) = Flux.logitcrossentropy(m(x),y)

# ╔═╡ ac248b60-00ce-11eb-3589-e5522bb8f0ea
optimiser = Descent(0.3)

# ╔═╡ b0f07cf8-00ce-11eb-37a0-59880c3055b4
Flux.train!(loss, Flux.params(m), train_data, optimiser)

# ╔═╡ b43b19c2-00ce-11eb-032f-11a2e2c82b91
accuracy_score = accuracy(X, labels, m)

# ╔═╡ Cell order:
# ╟─0eb7608c-00cc-11eb-0abe-bd2bca0d2b3e
# ╠═71c5500c-00cc-11eb-3162-a10318958fc2
# ╠═703d8bca-00cd-11eb-1b57-af64a482bd8a
# ╠═78ac553e-00cd-11eb-26b7-f5bdbe229dd7
# ╠═9ae94d2a-00cd-11eb-37b9-1dc46a16a49d
# ╠═a23e157c-00cd-11eb-268c-d992c466fdd8
# ╠═bd0aafe6-00cd-11eb-21ca-fd6c71a68f2f
# ╠═34ebec92-00d2-11eb-3f06-c30d260fddda
# ╠═afdc663e-00cd-11eb-04b7-899986c22aee
# ╠═c37e9a4a-00cd-11eb-32ce-b514071f2623
# ╠═cc6f4438-00cd-11eb-01c8-578b14b406b6
# ╠═df38d270-00cd-11eb-2899-419fbf880361
# ╠═2a87f08e-00d2-11eb-2889-1f2385032787
# ╠═44233b5c-00d2-11eb-3bc6-47248934072c
# ╠═50de7302-00d2-11eb-183a-752f712cbb38
# ╠═57182a38-00d2-11eb-1aa3-f39349334c9c
# ╠═7177531a-00d2-11eb-1f6c-53d12ebffb5c
# ╠═ffd9738e-00cd-11eb-0681-4d71a42001b1
# ╠═258db6e4-00ce-11eb-08c7-0ff693c39c95
# ╠═656fb6e0-00ce-11eb-014b-6d34bd579e77
# ╠═808a8b6c-00ce-11eb-27f7-fb9c543b53ff
# ╠═a24cb676-00ce-11eb-36cd-0f7f0e955ec1
# ╠═a7bb0414-00ce-11eb-35b6-e596f46224d3
# ╠═ac248b60-00ce-11eb-3589-e5522bb8f0ea
# ╠═b0f07cf8-00ce-11eb-37a0-59880c3055b4
# ╠═b43b19c2-00ce-11eb-032f-11a2e2c82b91
