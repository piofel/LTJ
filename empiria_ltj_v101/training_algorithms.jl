# Version: 0.8.4 (2019 February 22)
module TrainingAlgorithms

using Main.ConvolutionalNeuralNetwork: ConvModuleArch, ConvolutionalNetwork, conv_modules_signals, number_of_convolutional_modules, conv_module_output_dim, initial_convolutional_network_parameters
using Main.MathematicalFunctions: diag, matrix_to_vector, convolution, full_convolution
using Main.ConvNetMultilayerPerceptronHybrid: ConvPercHybrid, conv_mlp_hybrid_outputs, create_conv_mlp_hybrid
using Main.MultilayerPerceptron: MulLayerPerceptronArch, MulLayerPerceptron, mlp_outputs, mlp_signals, initial_multilayer_perceptron_parameters

using Random: shuffle

struct TrainingParameters
	# The learning rate for the multilayer perceptron
	mlp_learning_rate :: Float64
	# The convolutional network learning rate
	cnn_learning_rate :: Float64
	# The learning error tolerance
	tolerance :: Float64
	# A boundary to the number of training epochs
	training_epochs_num_bound :: Int64
	# A boundary to the number of training attempts
	training_attempts_num_bound :: Int64
	# An period (of training epochs) for checking the network error during training
	error_checking_period :: Int64
	# The method for choice of training pairs
	# - "sequential" - chooses all training pairs in natural order during training epoch
	# - "random" - chooses all training pairs in random order
	# - "random_part" - chooses only part of training pairs in training epoch in random order
	pair_choice_method :: String
	# Parameter for "random_part" selection of training pairs
	# In each training epoch, only this number of training pairs is selected
	training_subset_size :: Int64
	# Training method
	# - "backpropagation"
	# - "err_deriv_meas" (empirical measurement of error derivative, now obsolete and removed after version 0.8.1)
	train_method :: String
	# Intensity of displayed messages
	verbosity :: Int
	# Batch calculation of weight updates
	batching :: Bool
	# Momentum value 0-1
	momentum :: Float64
end

function squared_error(network::ConvPercHybrid,input_signals_desired_outputs_pair::Tuple{Matrix{Float64},Vector{Float64}})
        (input_signals,desired_outputs) = input_signals_desired_outputs_pair
        out = conv_mlp_hybrid_outputs(network,input_signals)
        e = desired_outputs .- out
        return transpose(e) * e
end

function squared_error(network::MulLayerPerceptron,input_signals_desired_outputs_pair::Tuple{Vector{Float64},Vector{Float64}})
        (input_signals,desired_outputs) = input_signals_desired_outputs_pair
        out = mlp_outputs(network,input_signals)
        e = desired_outputs .- out
        return transpose(e) * e
end

function mean_squared_error(network::Union{ConvPercHybrid,MulLayerPerceptron},training_set::Union{Vector{Tuple{Matrix{Float64},Vector{Float64}}},Vector{Tuple{Vector{Float64},Vector{Float64}}}})
		e :: Float64 = 0
		n = lastindex(training_set)
		for i = 1:n
			e = e + squared_error(network,training_set[i])
		end
		return e / n
end

struct MulLayerPerceptronErrorDerivatives
	derivatives_wrt_weights :: Vector{Matrix{Float64}}
	derivatives_wrt_biases :: Vector{Vector{Float64}}
end

function zero_error_derivatives(mlp::MulLayerPerceptron)
	n = lastindex(mlp.layers)
	derivs_wrt_weights = Vector{Matrix{Float64}}(undef,n)
	derivs_wrt_biases = Vector{Vector{Float64}}(undef,n)
	for i = 1:n
		(r,c) = size(mlp.layers[i].weights)
		derivs_wrt_weights[i] = zeros(r,c)
		l = lastindex(mlp.layers[i].biases)
		derivs_wrt_biases[i] = zeros(l) 
	end
	return MulLayerPerceptronErrorDerivatives(derivs_wrt_weights,derivs_wrt_biases)
end

struct ConvNetErrorDerivatives
	derivatives_wrt_filters :: Vector{Vector{Matrix{Float64}}}
	derivatives_wrt_biases :: Vector{Vector{Float64}}
end

function zero_error_derivatives(cnn::ConvolutionalNetwork)
	n = number_of_convolutional_modules(cnn)
	derivs_wrt_filters = Vector{Vector{Matrix{Float64}}}(undef,n)
	derivs_wrt_biases = Vector{Vector{Float64}}(undef,n)
	for i = 1:n
		m = conv_module_output_dim(cnn.convolutional_modules[i]) # number of filters
		mod_derivs_wrt_filters = Vector{Matrix{Float64}}(undef,m)
		for j = 1:m
			(r,c) = size(cnn.convolutional_modules[i].convolutional_layer.filters[j])
			mod_derivs_wrt_filters[j] = zeros(r,c)
		end
		l = length(cnn.convolutional_modules[i].convolutional_layer.biases)
		derivs_wrt_biases[i] = zeros(l)
		derivs_wrt_filters[i] = mod_derivs_wrt_filters
	end
	return ConvNetErrorDerivatives(derivs_wrt_filters,derivs_wrt_biases)
end

function bp_training_pair_presentation(network::ConvPercHybrid,training_pair::Tuple{Matrix{Float64},Vector{Float64}}) :: Tuple{MulLayerPerceptronErrorDerivatives,ConvNetErrorDerivatives}
	(input_signals,desired_outputs) = training_pair
	cnn_sig = conv_modules_signals(network.cnn.convolutional_modules,input_signals)
	cnn_out = last(cnn_sig.modules_outputs_list)
	cnn_out = matrix_to_vector(cnn_out)
	mlp_sig = mlp_signals(network.mlp,cnn_out)
	n = lastindex(network.mlp.layers)
	derivs_wrt_weights = Vector{Matrix{Float64}}(undef,n)
	derivs_wrt_biases = Vector{Vector{Float64}}(undef,n)
	l = network.mlp.layers[n]
	act = mlp_sig.activations[n]
	out = mlp_sig.outputs[n]
	afd = l.activation_function_derivative
	afdv = map(afd,act)
	afdm = diag(afdv)
	sens = -2 .* afdm * (desired_outputs - out) # sensitivity of the last layer
	derivs_wrt_weights[n] = sens * transpose(mlp_sig.outputs[n-1])
	derivs_wrt_biases[n] = sens
	for i = 1:(n-1)
		next_layer = l
		l = network.mlp.layers[n-i]
		act = mlp_sig.activations[n-i]
		out = mlp_sig.outputs[n-i]
		afd = l.activation_function_derivative
		afdv = map(afd,act)
		afdm = diag(afdv)
		sens = afdm * transpose(next_layer.weights) * sens
		if n-i != 1
			derivs_wrt_weights[n-i] = sens * transpose(mlp_sig.outputs[n-i-1])
		else
			derivs_wrt_weights[n-i] = sens * transpose(cnn_out)
		end
		derivs_wrt_biases[n-i] = sens
	end
	dedout = transpose(network.mlp.layers[1].weights) * sens # sensitivity of the error to mlp input (cnn output)
	mlp_err_derivs = MulLayerPerceptronErrorDerivatives(derivs_wrt_weights,derivs_wrt_biases)
	nm = number_of_convolutional_modules(network.cnn)
	derivs_wrt_filters = Vector{Vector{Matrix{Float64}}}(undef,nm)
	derivs_wrt_biases = Vector{Vector{Float64}}(undef,nm)
	for i = 0:(nm-1)
		cm = network.cnn.convolutional_modules[nm-i]
		clo = cnn_sig.conv_layer_outputs_list[nm-i]
		dpool = map(cm.pooling_layer.pooling_function_derivative,clo)
		a = cnn_sig.activations_list[nm-i]
		na = lastindex(a)
		d = Vector{Matrix{Float64}}(undef,na)
		dedfilt = Vector{Matrix{Float64}}(undef,na)
		mod_derivs_wrt_biases = Vector{Float64}(undef,na)
		if nm-i != 1
			cmi = cnn_sig.modules_outputs_list[nm-i-1] # convolutional module input
		else
			cmi = input_signals
		end
		for j = 1:na
			d[j] = map(cm.convolutional_layer.activation_function_derivative,a[j]) # d(conv. layer out) / d(convolution or bias)
			d[j] = dedout[j] * dpool[j] .* d[j] # d(net. squared error) / d(convolution or bias)
			dedfilt[j] = convolution(cmi,d[j]) # d(net. squared error) / d(filter)
			filter = cm.convolutional_layer.filters[j]
			bias = network.cnn.convolutional_modules[nm-i].convolutional_layer.biases[j]
			mod_derivs_wrt_biases[j] = sum(d[j])
		end
		derivs_wrt_filters[nm-i] = dedfilt
		derivs_wrt_biases[nm-i] = mod_derivs_wrt_biases
		if nm-i != 1
			dedout = zeros(size(cmi))
			for j = 1:na
				filter = cm.convolutional_layer.filters[j]
				dedout = dedout .+ full_convolution(d[j],rot180(filter)) # d(net. squared error) / d(module input or previous module output)
			end
		end
	end
	cnn_err_derivs = ConvNetErrorDerivatives(derivs_wrt_filters,derivs_wrt_biases)
	return (mlp_err_derivs,cnn_err_derivs)
end

function update_parameters(	net::ConvPercHybrid,
				training_parameters::TrainingParameters,
				mlp_err_derivs::MulLayerPerceptronErrorDerivatives,
				cnn_err_derivs::ConvNetErrorDerivatives,
				previous_mlp_err_derivs::MulLayerPerceptronErrorDerivatives,
				previous_cnn_err_derivs::ConvNetErrorDerivatives ) :: Tuple{ConvPercHybrid,MulLayerPerceptronErrorDerivatives,ConvNetErrorDerivatives}
	mlplr = training_parameters.mlp_learning_rate
	cnnlr = training_parameters.cnn_learning_rate
	mm = training_parameters.momentum
	cmm = 1 - mm
	n = lastindex(net.mlp.layers)
	for i = 1:n
		net.mlp.layers[i].weights -= cmm .* mlplr .* mlp_err_derivs.derivatives_wrt_weights[i]
		net.mlp.layers[i].weights -= mm .* mlplr .* previous_mlp_err_derivs.derivatives_wrt_weights[i]
		net.mlp.layers[i].biases -= cmm .* mlplr .* mlp_err_derivs.derivatives_wrt_biases[i]
		net.mlp.layers[i].biases -= mm .* mlplr .* previous_mlp_err_derivs.derivatives_wrt_biases[i]
	end
	n = number_of_convolutional_modules(net.cnn)
	for i = 1:n
		m = conv_module_output_dim(net.cnn.convolutional_modules[i]) # number of filters
		for j = 1:m
			net.cnn.convolutional_modules[i].convolutional_layer.filters[j] -= cmm .* cnnlr .* cnn_err_derivs.derivatives_wrt_filters[i][j]
			net.cnn.convolutional_modules[i].convolutional_layer.filters[j] -= mm .* cnnlr .* previous_cnn_err_derivs.derivatives_wrt_filters[i][j]
			net.cnn.convolutional_modules[i].convolutional_layer.biases[j] -= cmm .* cnnlr .* cnn_err_derivs.derivatives_wrt_biases[i][j]
			net.cnn.convolutional_modules[i].convolutional_layer.biases[j] -= mm .* cnnlr .* previous_cnn_err_derivs.derivatives_wrt_biases[i][j]
		end
	end
	previous_mlp_err_derivs = mlp_err_derivs
	previous_cnn_err_derivs = cnn_err_derivs
	return (net,previous_mlp_err_derivs,previous_cnn_err_derivs)
end

function training_epoch(	network :: ConvPercHybrid,
				training_set :: Vector{Tuple{Matrix{Float64},Vector{Float64}}},
				training_parameters :: TrainingParameters)
	if training_parameters.train_method == "backpropagation"
		tpp = bp_training_pair_presentation
	else
		error("Incorrect training method.")
	end
	if training_parameters.pair_choice_method == "sequential"
		ts = training_set
	elseif training_parameters.pair_choice_method == "random"
		ts = shuffle(training_set)
	elseif training_parameters.pair_choice_method == "random_part"
		ts = shuffle(training_set)
		n = lastindex(ts)
		if training_parameters.training_subset_size >= n
			ts = ts[1:n]
		else
			ts = ts[1:training_parameters.training_subset_size]
		end
	end
	previous_mlp_err_derivs = zero_error_derivatives(network.mlp)
	previous_cnn_err_derivs = zero_error_derivatives(network.cnn)
	n = lastindex(ts)
	if training_parameters.batching
		g = lastindex(network.mlp.layers)
		h = number_of_convolutional_modules(network.cnn)
    		(mlp_err_derivs,cnn_err_derivs) = tpp(network,ts[1])
		mlp_derivs_wrt_weights_ac = mlp_err_derivs.derivatives_wrt_weights
		mlp_derivs_wrt_biases_ac = mlp_err_derivs.derivatives_wrt_biases
		cnn_derivs_wrt_filters_ac = cnn_err_derivs.derivatives_wrt_filters
		cnn_derivs_wrt_biases_ac = cnn_err_derivs.derivatives_wrt_biases
    		for i = 2:n
    			(mlp_err_derivs,cnn_err_derivs) = tpp(network,ts[i])
			mlp_derivs_wrt_weights_ac .+= mlp_err_derivs.derivatives_wrt_weights
			mlp_derivs_wrt_biases_ac .+= mlp_err_derivs.derivatives_wrt_biases
			cnn_derivs_wrt_filters_ac .+= cnn_err_derivs.derivatives_wrt_filters
			cnn_derivs_wrt_biases_ac .+= cnn_err_derivs.derivatives_wrt_biases
		end
		mlp_derivs_wrt_weights_ac ./= n
		mlp_derivs_wrt_biases_ac ./= n
		cnn_derivs_wrt_filters_ac ./= n
		cnn_derivs_wrt_biases_ac ./= n
		mlp_err_derivs = MulLayerPerceptronErrorDerivatives(mlp_derivs_wrt_weights_ac,mlp_derivs_wrt_biases_ac)
		cnn_err_derivs = ConvNetErrorDerivatives(cnn_derivs_wrt_filters_ac,cnn_derivs_wrt_biases_ac)
		(network,previous_mlp_err_derivs,previous_cnn_err_derivs) = update_parameters(network,training_parameters,mlp_err_derivs,cnn_err_derivs,previous_mlp_err_derivs,previous_cnn_err_derivs)
	else
    		for i = 1:n
    			(mlp_err_derivs,cnn_err_derivs) = tpp(network,ts[i])
			(network,previous_mlp_err_derivs,previous_cnn_err_derivs) = update_parameters(network,training_parameters,mlp_err_derivs,cnn_err_derivs,previous_mlp_err_derivs,previous_cnn_err_derivs)
		end
    	end
    	return network
end

function train(	net :: ConvPercHybrid,
				training_set :: Vector{Tuple{Matrix{Float64},Vector{Float64}}},
				training_parameters :: TrainingParameters,
				training_attempt_number :: Int ) :: Tuple{ConvPercHybrid,Bool}
    	mse = mean_squared_error(net,training_set)
		err_meas_time = training_parameters.error_checking_period
		training_success = false
        i::Int = 0
        while true
			if mse <= training_parameters.tolerance
				training_success = true
        		if training_parameters.verbosity > 0
					println("Training finished with success.")
				end
				break
			end
            i = i + 1
            net = training_epoch(net,training_set,training_parameters)
        	if training_parameters.verbosity > 2
					msg = "Training attempt " * string(training_attempt_number) * ": training epoch " * string(i) * " completed."
					println(msg)
        	end
            if i >= training_parameters.training_epochs_num_bound
				if training_parameters.verbosity > 0
					println("Max. allowed number of training epochs reached.")
				end
				break
			end
			if i == err_meas_time
    				mse = mean_squared_error(net,training_set)
        			if training_parameters.verbosity > 2
					msg = "Mean squared error: "
					msg *= string(mse)
					println(msg)
				end
				err_meas_time = err_meas_time + training_parameters.error_checking_period
			end
        end
		return  (net,training_success)
end

function auto_develop_network(	cnn_arch::Vector{ConvModuleArch},
								mlp_arch::MulLayerPerceptronArch,
								training_set::Vector{Tuple{Matrix{Float64},Vector{Float64}}},
								training_parameters::TrainingParameters,
								init_parameters_lower_bound::Float64,
								init_parameters_upper_bound::Float64 )
        if training_parameters.verbosity > 0
            println("Training in progress, please wait.")
        end
    	cnn_parameters = initial_convolutional_network_parameters(cnn_arch,init_parameters_lower_bound,init_parameters_upper_bound)
    	mlp_parameters = initial_multilayer_perceptron_parameters(mlp_arch,init_parameters_lower_bound,init_parameters_upper_bound)
    	net = create_conv_mlp_hybrid(cnn_parameters,mlp_parameters)
		h::Int = 0
		while h < training_parameters.training_attempts_num_bound
			h = h + 1
			(net,training_success) = train(net,training_set,training_parameters,h)
			if training_success
				break
			end
    		cnn_parameters = initial_convolutional_network_parameters(cnn_arch,init_parameters_lower_bound,init_parameters_upper_bound)
    		mlp_parameters = initial_multilayer_perceptron_parameters(mlp_arch,init_parameters_lower_bound,init_parameters_upper_bound)
    		net = create_conv_mlp_hybrid(cnn_parameters,mlp_parameters)
		end
        if training_parameters.verbosity > 1
			println("Total number of training attempts: " * string(h))
        end
        return net
end

end  # module TrainingAlgorithms
