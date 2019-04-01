include("mathematical_functions.jl")
include("multilayer_perceptron.jl")
include("convolutional_neural_network.jl")
include("conv_perc_hybrid.jl")
include("training_algorithms.jl")
include("parameters_preservation.jl")
include("language_technology.jl")
include("preprocessing.jl")

using Main.MultilayerPerceptron: MulLayerPerceptronArch, MulLayerPerceptronParameters, initial_multilayer_perceptron_parameters, create_mlp, mlp_outputs
using Main.TrainingAlgorithms: train, auto_develop_network, TrainingParameters, squared_error, auto_develop_network
using Main.ConvolutionalNeuralNetwork: ConvModuleArch, initial_convolutional_network_parameters, conv_module_output_dim
using Main.ConvNetMultilayerPerceptronHybrid: create_conv_mlp_hybrid, conv_mlp_hybrid_outputs
using Main.ParametersPreservation: save_network, load_conv_mlp_hybrid
using Main.Preprocessing: training_set_files_from_sick, load_training_set, training_set_files_from_hs_codes_1, transform_hs_codes_database_1
using Main.LanguageTechnology: compare_meaning, text_matrix_width, text_matrix

	# The learning rate for the multilayer perceptron
const	learning_rate = 1.0
	# The convolutional network learning rate
const	cnn_learning_rate = 1.0
	# The learning error tolerance
const	tolerance = 0.0008
	# A boundary to the number of training epochs
const	training_epochs_num_bound = 5000
	# A boundary to the number of training attempts
const	training_attempts_num_bound = 1000
	# An period (of training epochs) for checking the network error during training
const	error_checking_period = 5
	# The method for choice of training pairs
	# - "sequential" - chooses all training pairs in natural order during training epoch
	# - "random" - chooses all training pairs in random order
	# - "random_part" - chooses only part of training pairs in training epoch in random order
const	pair_choice_method = "sequential"
	# Parameter for "random_part" selection of training pairs
	# In each training epoch, only this number of training pairs is selected
const	training_subset_size = 100
	# Training method
	# - "backpropagation"
	# - "err_deriv_meas" (empirical measurement of error derivative)
const	train_method = "backpropagation"
const 	verbosity = 3
	# Batch calculation of weight updates
const batching = true
	# Momentum value 0-1
const momentum = 0.9

const training_params = TrainingParameters(learning_rate,cnn_learning_rate,tolerance,training_epochs_num_bound,training_attempts_num_bound,error_checking_period,pair_choice_method,training_subset_size,train_method,verbosity,batching,momentum)

# Bounds to initially random parameters
const init_param_lower_bound = -1.0
const init_param_upper_bound = 1.0

const username = "karpio"
const intermediate_path = "Piotr"
# const username = "piotrfelisiak"
# const intermediate_path = "."
const saved_mlp_dir = "saved_mlp/"
const saved_cnn_dir = "saved_cnn/"
const empiria_path = "/home/$username/$intermediate_path/Workspace/research/empiria_ltj/"
const training_set_path = "/home/$username/$intermediate_path/Data/test_training_set_8/"
const training_set_dir = "training_set/"
const separator_file_name = "separator.txt"
const hs_codes_database_path = "/home/$username/$intermediate_path/Data/test_database_8/"
const hs_codes_database_file_name = "EN_HS_CODE_2005.tshsc"

const sparse_input = true

function test_1()
    c = [10 2; 3.5 0; 50 6.3; 7 8; 5.23 -9.2; 12.78 -0.45]
    d = [3 4; 4.5 -3; 6 7.3; 8 9; 1.23 -0.2; 1.78 0.5]
    des_out_1 = [0.0,1.0]
    des_out_2 = [1.0,0.0]
    tr_pair_1 = (c,des_out_1)
    tr_pair_2 = (d,des_out_2)
    ts = [tr_pair_1,tr_pair_2]
    arch_cm_1 = ConvModuleArch(2,[1,2,3,4],4,"tanh","one_max")
    arch_cm_2 = ConvModuleArch(1,[2,3,4],5,"tanh","one_max")
    cnn_arch = [arch_cm_1,arch_cm_2]
    mlp_arch = MulLayerPerceptronArch(15,[20,30,2],["sigmoidal","sigmoidal","sigmoidal"])
    net = auto_develop_network(cnn_arch,mlp_arch,ts,training_params,init_param_lower_bound,init_param_upper_bound)
    if false
    	save_network(net,empiria_path,saved_cnn_dir,saved_mlp_dir)
    	net = load_conv_mlp_hybrid(empiria_path,saved_cnn_dir,saved_mlp_dir)
    end
    println("Outputs (for matrix c) after training: ")
    println(conv_mlp_hybrid_outputs(net,c))
    println("Outputs (for matrix d) after training: ")
    println(conv_mlp_hybrid_outputs(net,d))
end

function test_2()
    training_set_files_from_sick("/home/$username/$intermediate_path/Data/test_database_2/",training_set_path,training_set_dir,separator_file_name,verbosity)
end

function test_3()
	training_set_files_from_hs_codes_1(hs_codes_database_path,"EN_HS_CODE_2005.tdhsc",training_set_path,training_set_dir,separator_file_name,verbosity)
end

function test_4()
    #d = text_matrix_width()
    d=1
    arch_cm_1 = ConvModuleArch(d,[7,10],500,"tanh","one_max")
    arch_cm_2 = ConvModuleArch(1,[3,5],500,"tanh","one_max")
    cnn_arch = [arch_cm_1,arch_cm_2]
    cnn_out_dim = conv_module_output_dim(last(cnn_arch))
    mlp_arch = MulLayerPerceptronArch(cnn_out_dim,[1000,1],["sigmoidal","sigmoidal","sigmoidal"])
    ts = load_training_set(training_set_path,training_set_dir,sparse_input)
    net = auto_develop_network(cnn_arch,mlp_arch,ts,training_params,init_param_lower_bound,init_param_upper_bound)
    save_network(net,empiria_path,saved_cnn_dir,saved_mlp_dir)
end

function test_5()
	transform_hs_codes_database_1(hs_codes_database_path,"EN_HS_CODE_2005.csv",hs_codes_database_path,"EN_HS_CODE_2005.tdhsc")
end
