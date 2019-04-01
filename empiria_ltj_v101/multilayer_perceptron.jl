module MultilayerPerceptron

using Main.MathematicalFunctions: sigmoid, sigmoid_derivative, rectifier, heaviside_function, sym_binary, id, id_derivative, random_array, random_vector

mutable struct Layer
        weights :: Matrix{Float64}
        biases :: Vector{Float64}
        activation_function :: Function
		activation_function_derivative :: Function
        layer_type :: String
end

struct MulLayerPerceptron
        layers :: Vector{Layer}
end

struct MulLayerPerceptronParameters
        weights :: Vector{Matrix{Float64}}
        biases :: Vector{Vector{Float64}}
        layer_types:: Vector{String}
end

struct MulLayerPerceptronArch
	number_of_inputs :: Int
	numbers_of_neurons :: Vector{Int}
	layer_types :: Vector{String}
end

# Creates a multilayer perceptron from a list of weight matrices, corresponding bias vectors and layer types
function create_mlp(mlp_parameters::MulLayerPerceptronParameters)
        weight_matrices_list = mlp_parameters.weights
        bias_vectors_list = mlp_parameters.biases
        layer_types = mlp_parameters.layer_types
        n = lastindex(weight_matrices_list)
        layers = Array{Layer,1}(undef,n)
        for i = 1:n
                if layer_types[i] == "relu"
                        actfun = rectifier
			actfunderiv = heaviside_function
                elseif layer_types[i] == "binary"
                        actfun = heaviside_function
			actfunderiv = none
                elseif layer_types[i] == "sym_binary"
                        actfun = sym_binary
			actfunderiv = none
                elseif layer_types[i] == "sigmoidal"
                        actfun = sigmoid
			actfunderiv = sigmoid_derivative
                elseif layer_types[i] == "linear"
                        actfun = id
			actfunderiv = id_derivative
                else
                        error("Incorrect network type.")
                end
                layers[i] = Layer(weight_matrices_list[i],bias_vectors_list[i],actfun,actfunderiv,layer_types[i])
        end
        return MulLayerPerceptron(layers)
end

# Gives initial, random parameters (weights and biases) and layer types for the neural network from a number of inputs and a list of numbers of neurons in consecutive layers, e.g. [3,4,2], and a list of layer types
function initial_multilayer_perceptron_parameters(mlp_arch::MulLayerPerceptronArch,init_parameters_lower_bound::Float64,init_parameters_upper_bound::Float64) :: MulLayerPerceptronParameters
        l = lastindex(mlp_arch.numbers_of_neurons)
        weights = Array{Array{Float64,2},1}(undef,l)
        biases = Array{Array{Float64,1},1}(undef,l)
        ncols = mlp_arch.number_of_inputs
        for i = 1:l
				n = mlp_arch.numbers_of_neurons[i]
                w = random_array(n,ncols,init_parameters_lower_bound,init_parameters_upper_bound)
                b = random_vector(n,init_parameters_lower_bound,init_parameters_upper_bound)
                weights[i] = w
                biases[i] = b
                ncols = n
        end
        return MulLayerPerceptronParameters(weights,biases,mlp_arch.layer_types)
end

function perceptron_activations(layer::Layer,input_signals::Array{Float64,1}) :: Array{Float64,1}
    prod = layer.weights * input_signals
    return prod + layer.biases
end

function perceptron_outputs(layer::Layer,activations::Array{Float64,1}) :: Array{Float64,1}
        return map(layer.activation_function,activations)
end

function mlp_outputs(network::MulLayerPerceptron,input_signals::Vector{Float64}) :: Vector{Float64}
        l = network.layers
        n = lastindex(l)
        s = input_signals
        for i = 1:n
                a = perceptron_activations(l[i],s)
                s = perceptron_outputs(l[i],a)
        end
        return s
end

struct MulLayerPerceptronSignals
	activations :: Array{Array{Float64,1},1}
	outputs :: Array{Array{Float64,1},1}
end

function mlp_signals(network::MulLayerPerceptron,input_signals::Vector{Float64})
	n = lastindex(network.layers)
	out = Array{Array{Float64,1},1}(undef,n)
	act = Array{Array{Float64,1},1}(undef,n)
	input = input_signals
	for i = 1:n
		l = network.layers[i]
		act[i] = perceptron_activations(l,input)
		out[i] = perceptron_outputs(l,act[i])
		input = out[i] # for the next layer
	end
	return MulLayerPerceptronSignals(act,out)
end

end  # module
