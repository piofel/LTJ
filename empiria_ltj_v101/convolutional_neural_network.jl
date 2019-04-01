module ConvolutionalNeuralNetwork

using Main.MathematicalFunctions: random_array, random_vector, tanh_derivative, maximum_fun_derivative, convolution

struct ConvolutionalLayer
    filters :: Vector{Matrix{Float64}}
    biases :: Vector{Float64}
    activation_function :: Function
    activation_function_derivative :: Function
	activation_function_type :: String
end

struct PoolingLayer
    pooling_function :: Function
    pooling_function_derivative :: Function
    pooling_function_type :: String
end

struct ConvolutionalModule
    convolutional_layer :: ConvolutionalLayer
    pooling_layer :: PoolingLayer
end

struct ConvolutionalNetwork
    convolutional_modules :: Vector{ConvolutionalModule}
end

# A data structure for description of architecture of a convolutional module
struct ConvModuleArch
    filters_width :: Int
    region_sizes :: Array{Int,1} # A list of heights of the filters
    filters_per_region_size :: Int
    activation_function_type :: String
    pooling_function_type :: String
end

struct ConvModuleParameters
    filters :: Vector{Matrix{Float64}}
    biases :: Vector{Float64}
    activation_function_type :: String
    pooling_function_type :: String
end

function number_of_convolutional_modules(network::ConvolutionalNetwork)
	lastindex(network.convolutional_modules)
end

# Number of outputs of convolutional module (equal to number of filters)
function conv_module_output_dim(cm::ConvolutionalModule)
	return lastindex(cm.convolutional_layer.filters)
end

# Number of outputs of convolutional module calculated from module architecture
function conv_module_output_dim(arch::ConvModuleArch)
    nrs = lastindex(arch.region_sizes)
    fpr = arch.filters_per_region_size
	return nrs * fpr
end

function initial_conv_module_parameters(arch::ConvModuleArch,init_parameters_lower_bound::Float64,init_parameters_upper_bound::Float64) :: ConvModuleParameters
	nf = conv_module_output_dim(arch)
    f = Vector{Matrix{Float64}}(undef,nf)
    b = Vector{Float64}(undef,nf)
    nrs = lastindex(arch.region_sizes)
    fpr = arch.filters_per_region_size
    k = 1 :: Int
    for i = 1:nrs
	    for j = 1:fpr
			f[k] = random_array(arch.region_sizes[i],arch.filters_width,init_parameters_lower_bound,init_parameters_upper_bound)
			b[k] = random_vector(1,init_parameters_lower_bound,init_parameters_upper_bound)[1]
			k = k + 1
	    end
    end
	return ConvModuleParameters(f,b,arch.activation_function_type,arch.pooling_function_type)
end

function initial_convolutional_network_parameters(archv::Vector{ConvModuleArch},init_parameters_lower_bound::Float64,init_parameters_upper_bound::Float64) :: Vector{ConvModuleParameters}
    ncm = lastindex(archv)
    cnn_parameters = Vector{ConvModuleParameters}(undef,ncm)
    for i = 1:ncm
        cnn_parameters[i] = initial_conv_module_parameters(archv[i],init_parameters_lower_bound::Float64,init_parameters_upper_bound::Float64)
    end
	return cnn_parameters
end

function create_convolutional_module(params::ConvModuleParameters)
    if params.activation_function_type == "relu"
        af = rectifier
		afd = heaviside_function
	elseif params.activation_function_type == "tanh"
	    af = tanh
		afd = tanh_derivative
    elseif params.activation_function_type == "sigmoidal"
		af = sigmoid
		afd = sigmoid_derivative
    else
        error("Incorrect activation function type.")
    end
    if params.pooling_function_type == "one_max"
        pf = maximum
		pfd = maximum_fun_derivative
    else
        error("Incorrect pooling function type.")
    end
    conv_layer = ConvolutionalLayer(params.filters,params.biases,af,afd,params.activation_function_type)
    pool_layer = PoolingLayer(pf,pfd,params.pooling_function_type)
    return ConvolutionalModule(conv_layer,pool_layer)
end

function create_conv_network(paramsv::Vector{ConvModuleParameters})
    ncm = lastindex(paramsv)
    conv_modules = Vector{ConvolutionalModule}(undef,ncm)
    for i = 1:ncm
		conv_modules[i] = create_convolutional_module(paramsv[i])
    end
    return ConvolutionalNetwork(conv_modules)
end

function add_bias(input_matrix::Matrix{Float64},bias::Float64)
	(r,c) = size(input_matrix)
	input_matrix + bias * ones(r,c)
end

function conv_layer_activations(filters::Vector{Matrix{Float64}},biases::Vector{Float64},input_matrix::Matrix{Float64}) :: Vector{Matrix{Float64}}
	n = lastindex(filters)
	activ = Vector{Matrix{Float64}}(undef,n)
	for i = 1:n
		c = convolution(input_matrix,filters[i])
		activ[i] = add_bias(c,biases[i])
	end
	return activ
end

function conv_layer_activations(layer::ConvolutionalLayer,input_matrix::Matrix{Float64}) :: Vector{Matrix{Float64}}
	return conv_layer_activations(layer.filters,layer.biases,input_matrix)
end

function conv_layer_outputs(layer::ConvolutionalLayer,layer_activations::Vector{Matrix{Float64}}) :: Vector{Matrix{Float64}}
	l = lastindex(layer_activations)
	out = Vector{Matrix{Float64}}(undef,l)
	for h = 1:l
		(n,m) = size(layer_activations[h])
		o = Matrix{Float64}(undef,n,m)
		for j = 1:m
			for i = 1:n
				o[i,j] = layer.activation_function(layer_activations[h][i][j])
			end
		end
		out[h] = o
	end
	return out
end

function pool_layer_outputs(layer::PoolingLayer,input_matrices::Vector{Matrix{Float64}}) :: Matrix{Float64}
	n = lastindex(input_matrices)
	out = Matrix{Float64}(undef,n,1)
	for i = 1:n
		out[i] = layer.pooling_function(input_matrices[i])
	end
	return out
end

function conv_net_outputs(network::ConvolutionalNetwork,input_matrix::Matrix{Float64}) :: Matrix{Float64}
	cnn_sig = conv_modules_signals(network.convolutional_modules,input_matrix)
	return last(cnn_sig.modules_outputs_list)
end

struct ConvModulesSignals
	activations_list :: Vector{Vector{Matrix{Float64}}}
	conv_layer_outputs_list :: Vector{Vector{Matrix{Float64}}}
	modules_outputs_list :: Vector{Matrix{Float64}}
end

function conv_modules_signals(modules::Vector{ConvolutionalModule},input_matrix::Matrix{Float64})
	n = lastindex(modules)
	activ = Vector{Vector{Matrix{Float64}}}(undef,n)
	cvl_out = Vector{Vector{Matrix{Float64}}}(undef,n)
	mod_out = Vector{Matrix{Float64}}(undef,n)
	inp = input_matrix
	for i = 1:n
		cl = modules[i].convolutional_layer
		a = conv_layer_activations(cl,inp)
		activ[i] = a
		clo = conv_layer_outputs(cl,a)
		cvl_out[i] = clo
		mo = pool_layer_outputs(modules[i].pooling_layer,clo)
		mod_out[i] = mo
		inp = mo
	end
	ConvModulesSignals(activ,cvl_out,mod_out)
end

end  # module ConvolutionalNeuralNetwork
