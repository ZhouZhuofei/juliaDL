### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ b438ab92-fefd-11ea-1054-632292d0965c
begin
	using Flux
	using Flux: @epochs
	using DataFrames
	using Gadfly
	using LinearAlgebra
	using Random
end

# ╔═╡ d6e28796-fefb-11ea-1b08-bbe1e9f98682
md"# 线性回归

## 基本

- 模型
- 训练
- 数据
- 损失函数
- 优化
***
"


# ╔═╡ a3a08c06-fefc-11ea-38d3-2526f9a1ccd2
md"$y = Xw + b$"

# ╔═╡ b3f67ca0-fefc-11ea-17cb-373174529e5c
md"
根据误差，利用梯度对参数$W$和$b$进行更新。

$W = W - \frac{lr}{batchsize} \frac{\partial L(W, b)}{\partial W}$

其中$batchsize$是每批数据训练大小，$lr$是学习率。"

# ╔═╡ 5a0fac6a-fefd-11ea-1daa-135ba73382ac
md"**例子**"

# ╔═╡ 924bbece-fefe-11ea-3cdf-75c5fe8eb392
begin
	#prepare data
	Random.seed!(123)
	num_inputs = 2
	num_inputs = 2
	num_examples = 1000
	True_w = [2.5, 3.0]
	True_b = 4.2
	features = randn(Float64, (num_examples, num_inputs))
	labels = features * True_w .+ True_b
	labels += randn(size(labels)) * 0.05
	regData = []
	for i in 1:num_examples
  		push!(regData, (features[i,:], labels[i]))
	end

end

# ╔═╡ 6293416a-feff-11ea-08f3-ab1625da01ab
x1 = reshape(features[:,1], (1, 1000))

# ╔═╡ 67ef56b2-feff-11ea-0d3a-5fcfab8cd1d8
x2 = reshape(features[:, 2], (1, 1000))

# ╔═╡ 00a23774-ff00-11ea-2e13-e96bf9905d60
#define Model
m = Chain(Dense(2,1))

# ╔═╡ 0632bd28-ff00-11ea-3f4f-214aaf6ecf73
#define Loss function
loss(x,y) = Flux.mse(m(x), y)

# ╔═╡ 1706aba0-ff00-11ea-12b7-159a1cc6ca0a
#opt
opt = Descent(0.3)


# ╔═╡ 32f1e2a8-ff00-11ea-2c18-95d2536c5068
@epochs 100 Flux.train!(loss, Flux.params(m), regData, opt)

# ╔═╡ 537447da-ff00-11ea-1c38-a30b89cc8c10
m([x1; x2])

# ╔═╡ 63707fde-ff00-11ea-2485-2b716b32b584
labels

# ╔═╡ c351b254-ff00-11ea-2b8a-677da5b21bd2
plot(x=m([x1;x2]), y=labels, Guide.xlabel("Predict Value"), Guide.ylabel("reality value"), Guide.title("Pre vs Real"))

# ╔═╡ 951be72a-ff01-11ea-1199-3dd0e9831890
md"
详细步骤：👇

```julia
batch_size = 10
#init w, b
w = rand(2)
b = rand(1)

#learning rate
lr = 0.3

#epoch
epoch = 3

linreg(X, w, b) = X*w .+ b
#define Loss
function Loss(ŷ, y)
	sum((ŷ - y) .^ 2)
end

function train_model(batch_size, features, labels, w, b, lr, epoch)
	dt_row, dt_col = size(features)
	indices = 1:dt_row
	Rond_indices = Random.shuffle(indices)
	for j in 1:epoch
		for i in range(1, dt_row, step = batch_size)
			j = Rond_indices[i:min(i+batch_size, dt_row)]
			X = features[j,:]
			y = labels[j,:]
			L = Loss(linreg(X, w, b), y)
			gs = gradient(() -> Loss(linreg(X, w, b), y), Flux.params(w, b))
			w -= lr/batch_size * gs[w]
			b -= lr/batch_size * gs[b]
		end
	end
	return w,b

end

ŵ, b̂ = train_model(batch_size, features, labels, w, b, lr, epoch)

pre_y = features * ŵ .+ b̂

print(pre_y)

```"

# ╔═╡ Cell order:
# ╟─d6e28796-fefb-11ea-1b08-bbe1e9f98682
# ╟─a3a08c06-fefc-11ea-38d3-2526f9a1ccd2
# ╟─b3f67ca0-fefc-11ea-17cb-373174529e5c
# ╟─5a0fac6a-fefd-11ea-1daa-135ba73382ac
# ╠═b438ab92-fefd-11ea-1054-632292d0965c
# ╠═924bbece-fefe-11ea-3cdf-75c5fe8eb392
# ╠═6293416a-feff-11ea-08f3-ab1625da01ab
# ╠═67ef56b2-feff-11ea-0d3a-5fcfab8cd1d8
# ╠═00a23774-ff00-11ea-2e13-e96bf9905d60
# ╠═0632bd28-ff00-11ea-3f4f-214aaf6ecf73
# ╠═1706aba0-ff00-11ea-12b7-159a1cc6ca0a
# ╠═32f1e2a8-ff00-11ea-2c18-95d2536c5068
# ╠═537447da-ff00-11ea-1c38-a30b89cc8c10
# ╠═63707fde-ff00-11ea-2485-2b716b32b584
# ╠═c351b254-ff00-11ea-2b8a-677da5b21bd2
# ╟─951be72a-ff01-11ea-1199-3dd0e9831890
