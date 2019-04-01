module LanguageTechnology

using Main.ConvNetMultilayerPerceptronHybrid: ConvPercHybrid, conv_mlp_hybrid_outputs

using DelimitedFiles: readdlm
using SparseArrays: sparse

struct CharVectors
	characters :: String
	char_vectors :: Matrix{Float64}
end

struct CharNumbers
	characters :: String
	char_numbers :: Vector{Float64}
end

const char_set = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz 0123456789()-'|;,./"

function create_char_vectors(list_of_characters::String) :: CharVectors
	n = lastindex(list_of_characters)
	m = zeros(n,n)
	for i = 1:n
		m[i,i] = 1.0
	end	
	return CharVectors(list_of_characters,m)
end

function create_text_m_representation(list_of_characters::String) :: CharNumbers
	n = lastindex(list_of_characters)
	m = Vector{Float64}(undef,n)
	d = 0.01
	m[1] = d
	for i = 2:n
		m[i] = m[i-1] + d
	end
	return CharNumbers(list_of_characters,m)
end

const char_vec = create_char_vectors(char_set)
const text_m_representation = create_text_m_representation(char_set)

function get_char_vector(char::Char)
	n = lastindex(char_vec.characters)
	for i = 1:n
		if char == char_vec.characters[i]
			return char_vec.char_vectors[:,i]
		end
	end
	println("Warning: the character '$char' has no vector representation.")
end

function get_char_number(char::Char) :: Float64
	n = lastindex(text_m_representation.characters)
	for i = 1:n
		if char == text_m_representation.characters[i]
			return text_m_representation.char_numbers[i]
		end
	end
	println("Warning: the character '$char' has no vector representation.")
end

function text_matrix(text::String)
    # d = text_matrix_width()
    d = 1
    n = lastindex(text)
    tm = zeros(n,d)
    for i = 1:n
	    # tm[i,:] = get_char_vector(text[i])
	    tm[i,:] = [get_char_number(text[i])]
    end
    # return sparse(tm)
    return tm
end

function text_matrix_width()
	(r,_) = size(char_vec.char_vectors)
	return r
end

# Creates separator, i.e. a block between two compared text matrices
function create_separator() :: Matrix{Float64}
	separator_height = 20
        # d = text_matrix_width()
	d = 1
        separator = zeros(separator_height,d)
	return separator
end

function compare_meaning(	text_a::String,
				text_b::String,
				net::ConvPercHybrid,
				training_set_path::String,
				training_set_dir::String,
				separator_file_name::String ) :: Float64
    text_a_num = text_matrix(text_a)
    text_b_num = text_matrix(text_b)
    path = training_set_path * training_set_dir
    separator = readdlm(path * separator_file_name, ',', Float64, '\n')
    input = vcat(text_a_num,separator,text_b_num)
    return conv_mlp_hybrid_outputs(net,input)[1]
end

function compare_meaning(	text_a_matrix::Matrix{Float64},
			 	text_b::String,
				net::ConvPercHybrid,
				training_set_path::String,
				training_set_dir::String,
				separator_file_name::String ) :: Float64
    text_b_num = text_matrix(text_b)
    path = training_set_path * training_set_dir
    separator = readdlm(path * separator_file_name, ',', Float64, '\n')
    input = vcat(text_a_matrix,separator,text_b_num)
    return conv_mlp_hybrid_outputs(net,input)[1]
end

end
