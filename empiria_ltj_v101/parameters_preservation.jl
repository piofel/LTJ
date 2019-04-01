module ParametersPreservation

using Main.MultilayerPerceptron: MulLayerPerceptron, MulLayerPerceptronParameters, create_mlp
using Main.ConvolutionalNeuralNetwork: ConvolutionalNetwork, ConvModuleParameters, number_of_convolutional_modules, conv_module_output_dim, create_conv_network
using Main.ConvNetMultilayerPerceptronHybrid: ConvPercHybrid
using Main.MathematicalFunctions: matrix_to_vector

using DelimitedFiles: writedlm, readdlm

function refresh_file(file_path::String) :: Nothing
	if isfile(file_path)
		rm(file_path)
	end
	return nothing
end

function refresh_dir(dir_path::String) :: Nothing
	if isdir(dir_path)
		rm(dir_path,recursive=true)
	end
	mkdir(dir_path)
	return nothing
end

function save_network(net::MulLayerPerceptron,empiria_path::String,saved_mlp_dir::String)
	path = empiria_path * saved_mlp_dir
	refresh_dir(path)
	activ_fun_file_name = "activation_functions.txt"
	n = lastindex(net.layers)
	for i = 1:n
		file_name = "weights_" * string(i) * ".txt"
		open(path * file_name, "w") do io
			writedlm(io, net.layers[i].weights, ',')
		end
		file_name = "biases_" * string(i) * ".txt"
		open(path * file_name, "w") do io
			writedlm(io, net.layers[i].biases, ',')
		end
		io = open(path * activ_fun_file_name, "a")
		write(io, net.layers[i].layer_type * "\n")
		close(io)
 	end
end

function save_network(net::ConvolutionalNetwork,empiria_path::String,saved_cnn_dir::String)
	path = empiria_path * saved_cnn_dir
	refresh_dir(path)
	activ_fun_file_name = "activation_functions.txt"
	pool_fun_file_name = "pooling_functions.txt"
	filter_num_file_name = "filter_numbers.txt"
	ncm = number_of_convolutional_modules(net)
	for i = 1:ncm
		cm = net.convolutional_modules[i]
		l = cm.convolutional_layer
		nf = conv_module_output_dim(cm)
		for j = 1:nf
			file_name = "conv_module_" * string(i) * "_filter_" * string(j) * ".txt"
			open(path * file_name, "w") do io
				writedlm(io, l.filters[j], ',')
			end
		end
		file_name = "conv_module_" * string(i) * "_biases.txt"
		open(path * file_name, "w") do io
			writedlm(io, l.biases, ',')
		end
		io = open(path * activ_fun_file_name,"a")
		write(io,l.activation_function_type * "\n")
		close(io)
		io = open(path * pool_fun_file_name,"a")
		write(io,cm.pooling_layer.pooling_function_type * "\n")
		close(io)
		io = open(path * filter_num_file_name,"a")
		write(io,string(nf) * "\n")
		close(io)
	end
end

function save_network(net::ConvPercHybrid,empiria_path::String,saved_cnn_dir::String,saved_mlp_dir::String)
	save_network(net.cnn,empiria_path,saved_cnn_dir)
	save_network(net.mlp,empiria_path,saved_mlp_dir)
end

function load_multilayer_perceptron(empiria_path::String,saved_mlp_dir::String) :: MulLayerPerceptron
	path = empiria_path * saved_mlp_dir
	activ_fun_file_name = "activation_functions.txt"
	af = readlines(path * activ_fun_file_name)
	n = lastindex(af)
	w = Vector{Matrix{Float64}}(undef,n)
	b = Vector{Vector{Float64}}(undef,n)
	for i = 1:n
		w[i] = readdlm(path * "weights_" * string(i) * ".txt", ',', Float64, '\n')
		bm = readdlm(path * "biases_" * string(i) * ".txt", ',', Float64, '\n')
		b[i] = matrix_to_vector(bm)
	end
	mlpp = MulLayerPerceptronParameters(w,b,af)
	return create_mlp(mlpp)
end

function load_convolutional_neural_network(empiria_path::String,saved_cnn_dir::String) :: ConvolutionalNetwork
	path = empiria_path * saved_cnn_dir
	activ_fun_file_name = "activation_functions.txt"
	pool_fun_file_name = "pooling_functions.txt"
	filter_num_file_name = "filter_numbers.txt"
	af = readlines(path * activ_fun_file_name)
	pf = readlines(path * pool_fun_file_name)
	fns = readlines(path * filter_num_file_name)
	ncm = lastindex(af)
	cnn_parameters = Vector{ConvModuleParameters}(undef,ncm)
	for i = 1:ncm
		nf = parse(Int,fns[i])
    	f = Vector{Matrix{Float64}}(undef,nf)
		for j = 1:nf
			f[j] = readdlm(path * "conv_module_" * string(i) * "_filter_" * string(j) * ".txt", ',', Float64, '\n')
		end
		b = readdlm(path * "conv_module_" * string(i) * "_biases.txt", ',', Float64, '\n')
		b = matrix_to_vector(b)
		cnn_parameters[i] = ConvModuleParameters(f,b,af[i],pf[i])
	end
	return create_conv_network(cnn_parameters)
end

function load_conv_mlp_hybrid(empiria_path::String,saved_cnn_dir::String,saved_mlp_dir::String)
	cnn = load_convolutional_neural_network(empiria_path,saved_cnn_dir)
	mlp = load_multilayer_perceptron(empiria_path,saved_mlp_dir)
	return ConvPercHybrid(cnn,mlp)
end

end
