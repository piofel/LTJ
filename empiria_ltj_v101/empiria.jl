# Top level file

include("mathematical_functions.jl")
include("multilayer_perceptron.jl")
include("convolutional_neural_network.jl")
include("conv_perc_hybrid.jl")
include("parameters_preservation.jl")
include("language_technology.jl")
include("preprocessing.jl")
include("teggs.jl")

using Main.Teggs: find_commodity
using Main.ParametersPreservation: load_conv_mlp_hybrid

const software_name = "Empiria LTJ"
const software_version = "1.0.1"
const software_version_date = software_version * " (from 2019 March 13)"

# const username = "karpio"
# const intermediate_path = "Piotr"
const username = "piotrfelisiak"
const intermediate_path = "."
const empiria_path = "/home/$username/$intermediate_path/Workspace/research/empiria_ltj/"
const saved_mlp_dir = "saved_mlp/"
const saved_cnn_dir = "saved_cnn/"
const training_set_path = "/home/$username/$intermediate_path/Data/"
const training_set_dir = "training_set/"
const separator_file_name = "separator.txt"
#const hs_codes_database_path = "/home/$username/$intermediate_path/Workspace/research/liat/HS_CODES/"
const hs_codes_database_path = "/home/$username/$intermediate_path/Data/test_database_4/"
const hs_codes_database_file_name = "EN_HS_CODE_2005.tdhsc"

function empiria_find_commodity()
	result_display_treshold = 0.333 # range 0-1
    	net = load_conv_mlp_hybrid(empiria_path,saved_cnn_dir,saved_mlp_dir)
	find_commodity(software_name,software_version,result_display_treshold,net,hs_codes_database_path,hs_codes_database_file_name,training_set_path,training_set_dir,separator_file_name)
end
