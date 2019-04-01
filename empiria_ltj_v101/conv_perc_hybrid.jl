module ConvNetMultilayerPerceptronHybrid

using Main.ConvolutionalNeuralNetwork: ConvolutionalNetwork, ConvModuleParameters, ConvModuleArch, create_conv_network, conv_module_output_dim, conv_net_outputs
using Main.MultilayerPerceptron: MulLayerPerceptron, MulLayerPerceptronParameters, create_mlp, mlp_outputs
using Main.MathematicalFunctions: matrix_to_vector

struct ConvPercHybrid
	cnn :: ConvolutionalNetwork
	mlp :: MulLayerPerceptron
end

function create_conv_mlp_hybrid(cnn_parameters::Vector{ConvModuleParameters},mlp_parameters::MulLayerPerceptronParameters)
	cnn = create_conv_network(cnn_parameters)
	cnn_out_dim = conv_module_output_dim(last(cnn.convolutional_modules))
	(_,mlp_in_dim) = size(first(mlp_parameters.weights))
	if cnn_out_dim == mlp_in_dim
		mlp = create_mlp(mlp_parameters)
		return ConvPercHybrid(cnn,mlp)
	else
		println("Convolutional network output dimension: ")
		print(cnn_out_dim)
		print("\n")
		println("Multilayer perceptron input dimentsion: ")
		print(mlp_in_dim)
		print("\n")
		error("Output dimension of the convolutional neural network does not match the input dimenstion of the multilayer perceptron.")
	end
end

function conv_mlp_hybrid_outputs(network::ConvPercHybrid,input_matrix::Matrix{Float64})
	cnn_out = conv_net_outputs(network.cnn,input_matrix)
	cnn_out = matrix_to_vector(cnn_out)
	mlp_outputs(network.mlp,cnn_out)
end

end
