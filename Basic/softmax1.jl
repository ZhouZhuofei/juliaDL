### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 545bb79c-fbf0-11ea-3455-67441d62f5fe
using Flux

# ╔═╡ 741102a8-fc05-11ea-2352-0bd39f49fcd3
using Statistics

# ╔═╡ 11f5a760-fbf4-11ea-286d-c9f6f30f83ed
using Plots

# ╔═╡ c4878102-fbf0-11ea-298e-eb205469990c
images = Flux.Data.FashionMNIST.images()

# ╔═╡ 01be0a34-fbf1-11ea-0beb-0521a2d634c4
labels = Flux.Data.FashionMNIST.labels()

# ╔═╡ 459673f4-fbf1-11ea-114a-53b01f61a81b
size(images[1])

# ╔═╡ 78f37170-fbf1-11ea-3dab-07eed2cba687
function get_fashion_label(labels)
	text_label = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
	return [text_label[i+1] for i in labels]
end

# ╔═╡ 05baaefc-fbf2-11ea-0a8b-29a1748f44d3
get_fashion_label(labels)

# ╔═╡ a5e15114-fbf3-11ea-1660-d3aaa45ff6c4
images[1]

# ╔═╡ 3e7e073c-fbf4-11ea-03b7-e723e9d36fe6
plot(images[1], title="short")

# ╔═╡ 93b8a5d6-fbf4-11ea-265e-5d6c49cc4242
m = Chain(Dense(784,10), softmax)

# ╔═╡ e007cc00-fbf4-11ea-32d8-f11128553f4f
size(images)

# ╔═╡ ce10f3f2-fc01-11ea-0c30-3945431c0fb2
Float64.(reshape(images[1],(784,1)))

# ╔═╡ 5d11b170-fc04-11ea-1c67-e7df917005f9
train_x = rand(784, 1000)

# ╔═╡ 76a58fd0-fc04-11ea-3ef0-5b61e18dac67
begin
	for i in 1:1000
		train_x[:,i] = Float64.(reshape(images[i],(784,1)))
	end
end

# ╔═╡ 8c989422-fc04-11ea-0a3c-d5ec2e7bc9dd
train_x

# ╔═╡ 9498eb48-fc01-11ea-24f5-c130e859c64f
get_fashion_label(labels)[1]

# ╔═╡ aff167a8-fc01-11ea-3e6c-5323f94be667
begin

	train_y = []
	for i in 1:1000

		push!(train_y, get_fashion_label(labels)[i])
	end
end
		

# ╔═╡ 1cdac5f0-fc02-11ea-1db6-67ec6ff27617
size(train_y)

# ╔═╡ 480a9cfc-fc03-11ea-0d4e-0d20b008513c
klasses = sort(unique(train_y))

# ╔═╡ c49d6140-fc04-11ea-12ed-612c4eafc3be
onehot_labels = Flux.onehotbatch(train_y, klasses)

# ╔═╡ 997dc9bc-fc04-11ea-2b6c-f1a54753f29a
train_data = Iterators.repeated((train_x, onehot_labels), 150)

# ╔═╡ acf6d226-fc02-11ea-3f18-493597b870d2
loss(x,y) = Flux.crossentropy(m(x), y)

# ╔═╡ 565b20cc-fc05-11ea-2306-53574dd81833
accuracy(x, y, model) = Flux.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# ╔═╡ c665f7c8-fc02-11ea-3eb8-9b0c019e5d87
opt = Descent(0.3)

# ╔═╡ df3dd4d0-fc04-11ea-0110-b3d5a5301326
Flux.train!(loss, Flux.params(m), train_data, opt)

# ╔═╡ 0bb69ca2-fc05-11ea-1ecf-3116ffb13f62
Flux.onecold(m(train_x[:,8]))

# ╔═╡ 34edea8c-fc05-11ea-1871-a36b2dc53a52
Flux.onecold(onehot_labels[:,8])

# ╔═╡ 5f892cac-fc05-11ea-247e-5d1a02a89a8a
accuracy(train_x, onehot_labels, m)

# ╔═╡ c351ba60-fc05-11ea-32c9-e5e8a65bc239
sum(Flux.onecold(m(train_x)) .== Flux.onecold(onehot_labels))

# ╔═╡ Cell order:
# ╠═545bb79c-fbf0-11ea-3455-67441d62f5fe
# ╠═741102a8-fc05-11ea-2352-0bd39f49fcd3
# ╠═c4878102-fbf0-11ea-298e-eb205469990c
# ╠═01be0a34-fbf1-11ea-0beb-0521a2d634c4
# ╠═459673f4-fbf1-11ea-114a-53b01f61a81b
# ╠═78f37170-fbf1-11ea-3dab-07eed2cba687
# ╠═05baaefc-fbf2-11ea-0a8b-29a1748f44d3
# ╠═a5e15114-fbf3-11ea-1660-d3aaa45ff6c4
# ╠═11f5a760-fbf4-11ea-286d-c9f6f30f83ed
# ╠═3e7e073c-fbf4-11ea-03b7-e723e9d36fe6
# ╠═93b8a5d6-fbf4-11ea-265e-5d6c49cc4242
# ╠═e007cc00-fbf4-11ea-32d8-f11128553f4f
# ╠═ce10f3f2-fc01-11ea-0c30-3945431c0fb2
# ╠═5d11b170-fc04-11ea-1c67-e7df917005f9
# ╠═76a58fd0-fc04-11ea-3ef0-5b61e18dac67
# ╠═8c989422-fc04-11ea-0a3c-d5ec2e7bc9dd
# ╠═9498eb48-fc01-11ea-24f5-c130e859c64f
# ╠═aff167a8-fc01-11ea-3e6c-5323f94be667
# ╠═997dc9bc-fc04-11ea-2b6c-f1a54753f29a
# ╠═1cdac5f0-fc02-11ea-1db6-67ec6ff27617
# ╠═480a9cfc-fc03-11ea-0d4e-0d20b008513c
# ╠═c49d6140-fc04-11ea-12ed-612c4eafc3be
# ╠═acf6d226-fc02-11ea-3f18-493597b870d2
# ╠═565b20cc-fc05-11ea-2306-53574dd81833
# ╠═c665f7c8-fc02-11ea-3eb8-9b0c019e5d87
# ╠═df3dd4d0-fc04-11ea-0110-b3d5a5301326
# ╠═0bb69ca2-fc05-11ea-1ecf-3116ffb13f62
# ╠═34edea8c-fc05-11ea-1871-a36b2dc53a52
# ╠═5f892cac-fc05-11ea-247e-5d1a02a89a8a
# ╠═c351ba60-fc05-11ea-32c9-e5e8a65bc239
