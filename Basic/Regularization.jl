### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 8e3d151e-0b8b-11eb-1f36-319328a7772c
using Flux

# ╔═╡ 92dfefce-0b8b-11eb-32b4-391aecbbb876
using Distributions

# ╔═╡ 54417ed0-0b8c-11eb-2064-230c0020f268
using IterTools: ncycle

# ╔═╡ 622d190a-0b8c-11eb-1f00-63d9bfdca449
using Random

# ╔═╡ 65e0990a-0b8c-11eb-01ce-233699db160b
using LinearAlgebra

# ╔═╡ a68b270e-0b8c-11eb-050e-b704f7137d1c
using Parameters: @with_kw

# ╔═╡ 20ef48ee-0b8e-11eb-3f94-313efbfc9478
using PlutoUI

# ╔═╡ 055b2642-0b8f-11eb-2777-c1159e9e0e21
using Plots

# ╔═╡ 0f73843e-0b90-11eb-1425-5dce556a16a9
using BenchmarkTools

# ╔═╡ c63890b6-0b8a-11eb-3771-5bde29bc994b
md"""
# Regularization

Weight attenuation is equivalent to regularization of $L_2$ norm, Regularization is a common method to deal with overfitting by adding penalty term to model loss function to make the learned model parameter values smaller.

$l(w, b) + \frac{\lambda}{2n}||w||^2$

among them，$||w||^2 = w_{1}^2 + w_{2}^{2}$

## Higer-dimensional linear regression experiments

Taking high dimensional linear regression as an example, an overfitting problem is introduced.The feature dimension of the data sample is $P$, and the set feature of the data of the training set and the test set is $x_1, x_2, ..., x_p$.

$y = 0.05 + \sum_{i=1}^{p} 0.01x_i + \epsilon$

so we set dimension $p = 200$, smaple number $n=20$:

"""

# ╔═╡ 81ed2efc-0b8b-11eb-1b99-33cb760768fb
md"""
- first import some packages:
"""

# ╔═╡ 6d6b9062-0b8c-11eb-1c2c-d1a65f7ff889
md"""
- do simulation:
"""

# ╔═╡ 949227d2-0b8c-11eb-3ce8-211d2b5010b1
begin
	@with_kw mutable struct Args
		n::Int = 20
		p::Int = 200
		lr::Float64 = 0.003
	end
	
	args = Args()
	
	w, b = ones(args.p)*0.01, 0.05
	features = rand(Normal(0,1), (120, args.p))
	labels = features * w .+ b + rand(120)/100
	train_data = Flux.Data.DataLoader((Array(features[1:20, :]'),Array(labels[1:20]')),batchsize=1)
	test_x,test_y = Array(features[21:120, :]'), Array(labels[21:120]')
end

# ╔═╡ 37796f14-0b8d-11eb-350a-cb297a03f349
begin
	train_loss = []
	test_loss = []
	for λ in 0:0.1:4
    	model = Chain(Dense(200,1))
    	ps = Flux.params(model)
    	opt = Descent(args.lr)
    	sqnorm(x) = sum(abs2, x)
    	loss(x,y) = Flux.mse(model(x), y) + λ*sum(sqnorm, Flux.params(model))
    	Flux.train!(loss, ps, train_data, opt);
    	push!(train_loss, Flux.mse(model(Array(features[1:20, :]')), Array(labels[1:20]')))
    	push!(test_loss, Flux.mse(model(test_x), test_y))
	end
	plot(0:0.1:4, [train_loss test_loss], label=["train_loss" "test_loss"], xlabel="λ", ylabel="Loss", title="regularization λ influence the loss")
end

# ╔═╡ 80566366-0b8f-11eb-3ee6-37f6064d46fe
md"""
As we can see, the coefficient of regularization is $\lambda$, if $0<\lambda<1$, we have high loss in test but at train set is great,absolutely, we overfitting train set. when the $\lambda$ become more larger, the test loss start to fall down.

## Dropout
In multilayer perceptron, we describe a multilayer perceptron with a single hidden layer, assuming that number of inputs is 4, and the number of hidden layer unit is 5. So


$h_i = \phi(x_1w_{1i} + x_2w_{2i} + x_3w_{3i}+x_4w_{4i} + b_i)$

among them, $\phi$ is activative function, $x_1,...x_4$ is input value, $w_{1i}, ..., w_{4i}$ is hidden layer parameters.

When use dropout way to this hidden layer, hidden units in this layer will have the chance to be dropped.set the probability is $p$. drop out don't change the expect.

Because in training process, hidden units will be dropped out by $p$, so output calculate can't rely too much on anyone in $h_1,..h_5$.

**For example:**
"""

# ╔═╡ ebc4d248-0b8f-11eb-1123-810b0d1586ab
md"**_Import data:_**"

# ╔═╡ 38c61114-0b90-11eb-3c6f-2d622821ce69
begin
	images = Flux.Data.FashionMNIST.images()
	labels_i = Flux.Data.FashionMNIST.labels()
	k = sort(unique(labels_i))
	onehot_labels = Flux.onehotbatch(labels_i, k)
	train_x = rand(784, 6000)
	for i in 1:6000
		train_x[:,i] = Float64.(reshape(images[i],(784,1)))
	end
	train_data_i = Flux.Data.DataLoader(Array(train_x[:, 1:4000]), Array(onehot_labels[:, 1:4000]), batchsize=10, shuffle=true)
end

# ╔═╡ 8fdaa60c-0b90-11eb-0fca-510bd890f89e
md"**_dropout some layers_**:
every layers has the same probability $p$ of being dropped."

# ╔═╡ d7dc8f6a-0b90-11eb-371b-d9be51947ec4
begin
	m_d = Chain(
    Dense(784, 256, relu),
    Dropout(0.2), 
    Dense(256, 256, relu), 
    Dropout(0.5), 
    Dense(256, 10, relu))
	loss1(x, y) = Flux.logitcrossentropy(m_d(x), y)
	opt_d = Descent(0.05)
	ps_d = Flux.params(m_d)
	@btime Flux.train!(loss1, ps_d, ncycle(train_data_i, 10), opt_d)
end

# ╔═╡ 850218e0-0b91-11eb-0fd9-f13ca40fcb84
md"define accuracy function:"

# ╔═╡ 9f6b4ccc-0b91-11eb-09f4-eb564e81dfc1
accuracy(x, y, model) = Flux.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# ╔═╡ b9dc1886-0b91-11eb-1bc3-9bb26f25af96
md"show the train accuracy:"

# ╔═╡ aa557fa6-0b91-11eb-0743-233eff7ac496
accuracy(Array(train_x[:, 1:4000]), Array(onehot_labels[:, 1:4000]), m_d)

# ╔═╡ d3021716-0b91-11eb-012c-2f69cb71d076
md"show the test accuracy:"

# ╔═╡ e6b006d0-0b91-11eb-2d8d-29558ea1d719
accuracy(Array(train_x[:, 4000:6000]), Array(onehot_labels[:, 4000:6000]), m_d)

# ╔═╡ Cell order:
# ╟─c63890b6-0b8a-11eb-3771-5bde29bc994b
# ╟─81ed2efc-0b8b-11eb-1b99-33cb760768fb
# ╠═8e3d151e-0b8b-11eb-1f36-319328a7772c
# ╠═92dfefce-0b8b-11eb-32b4-391aecbbb876
# ╠═54417ed0-0b8c-11eb-2064-230c0020f268
# ╠═622d190a-0b8c-11eb-1f00-63d9bfdca449
# ╠═65e0990a-0b8c-11eb-01ce-233699db160b
# ╠═a68b270e-0b8c-11eb-050e-b704f7137d1c
# ╠═20ef48ee-0b8e-11eb-3f94-313efbfc9478
# ╠═055b2642-0b8f-11eb-2777-c1159e9e0e21
# ╠═0f73843e-0b90-11eb-1425-5dce556a16a9
# ╟─6d6b9062-0b8c-11eb-1c2c-d1a65f7ff889
# ╠═949227d2-0b8c-11eb-3ce8-211d2b5010b1
# ╠═37796f14-0b8d-11eb-350a-cb297a03f349
# ╟─80566366-0b8f-11eb-3ee6-37f6064d46fe
# ╟─ebc4d248-0b8f-11eb-1123-810b0d1586ab
# ╠═38c61114-0b90-11eb-3c6f-2d622821ce69
# ╟─8fdaa60c-0b90-11eb-0fca-510bd890f89e
# ╠═d7dc8f6a-0b90-11eb-371b-d9be51947ec4
# ╠═850218e0-0b91-11eb-0fd9-f13ca40fcb84
# ╟─9f6b4ccc-0b91-11eb-09f4-eb564e81dfc1
# ╟─b9dc1886-0b91-11eb-1bc3-9bb26f25af96
# ╠═aa557fa6-0b91-11eb-0743-233eff7ac496
# ╠═d3021716-0b91-11eb-012c-2f69cb71d076
# ╠═e6b006d0-0b91-11eb-2d8d-29558ea1d719
