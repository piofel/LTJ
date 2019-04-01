module Preprocessing

using Main.MathematicalFunctions: matrix_to_vector
using Main.ParametersPreservation: refresh_dir, refresh_file
using Main.LanguageTechnology: create_separator, text_matrix

using DelimitedFiles: writedlm, readdlm
using SparseArrays: sparse

struct HsCode1
	symbol :: String
	description_level_1 :: String
	description_level_2 :: String
	description_level_3 :: String
	description_level_4 :: String
	description_level_5 :: String
end

const fixed_input_matrix_height = 400

# Fix anomalous records and split regular records into fields
function fix_broken_and_split_1(database::Vector{String},database_record_length::Int64,field_separator::String) :: Vector{Vector{SubString{String}}}
	try_to_fix = true
	number_of_header_lines = 1
	n = lastindex(database)
	odb :: Vector{Vector{SubString{String}}} = [split(database[1],field_separator)]
	for i = (number_of_header_lines+1):n
		r :: Vector{SubString{String}} = split(database[i],field_separator)
		if lastindex(r) == database_record_length
			push!(odb,r)
		else
			msg = "Warning: an anomalous record found in the database; record no.:"
			msg *= string(i-number_of_header_lines) * "."
			if try_to_fix
				msg *= " An attempt to fix it."
				lr :: Vector{SubString{String}} = pop!(odb)
				desc :: SubString{String} = pop!(lr)
				desc = desc * " " * database[i] # extending description of previous record with the anomalous record
				push!(lr,desc)
				push!(odb,lr)
			end
			println(msg)
		end
	end
	return odb 
end

# Transforms a HS codes database from Microsoft Excel format into Tuidian format
# Suited for EN_HS_CODE_2005
function transform_hs_codes_database_1(	hs_codes_database_path::String,
					hs_codes_database_file_name::String,
					new_td_hs_codes_database_path::String,
					new_td_hs_codes_database_file_name::String ) :: Nothing
	source_database_record_length = 7	
	number_of_hierarchy_levels = 5
	field_separator = "\t"
	field_holder = "-"
	header = "RECORD_ID" * field_separator * "HS_CODE" * field_separator * "DESC_LVL_1" * field_separator * "DESC_LVL_2" * field_separator * "DESC_LVL_3" * field_separator * "DESC_LVL_4" * field_separator * "DESC_LVL_5"
	filepn = new_td_hs_codes_database_path * new_td_hs_codes_database_file_name
	refresh_file(filepn)
	io = open(filepn, "a")
	write(io, header * "\n")
	db = load_file_to_str_vector(hs_codes_database_path,hs_codes_database_file_name)
	db = fix_broken_and_split_1(db,source_database_record_length,field_separator)
	n = lastindex(db)
	desc_stack :: Vector{Tuple{SubString{String},Int64}} = []
	k :: Int64 = 0
	for h = 1:n
		if h > 1
			sl = db[h] 
			hierarchy_pos = parse(Int64,sl[5])
			if h < n
				nsl = db[h+1]
				next_hierarchy_pos = parse(Int64,nsl[5])
			else
				next_hierarchy_pos = hierarchy_pos
			end
			desc = sl[7]
			push!(desc_stack,(desc,hierarchy_pos))
			if hierarchy_pos >= next_hierarchy_pos
				k += 1
				hsc = sl[1]
				line_to_write = string(k) * field_separator
				line_to_write *= hsc * field_separator
				nd = lastindex(desc_stack)
				for i = 1:number_of_hierarchy_levels
					dt = field_holder
					for j = 1:nd
						(d,hp) = desc_stack[j]	
						if i == hierarchy_positions_map_1(hp)
							dt = d
						end
					end
					line_to_write *= dt * field_separator
				end
				write(io, line_to_write * "\n")
				if hierarchy_pos == next_hierarchy_pos
					pop!(desc_stack)
				end
				if hierarchy_pos > next_hierarchy_pos
					while !isempty(desc_stack)
						(_,hp) = pop!(desc_stack)
						if hp == next_hierarchy_pos
							break
						end
					end
				end
			end
		end
	end
	close(io)
	return nothing
end

# Suited for EN_HS_CODE_2005
function hierarchy_positions_map_1(hierarchy_pos::Int64) :: Int64
	if hierarchy_pos == 2
		return 1
	elseif hierarchy_pos == 4
		return 2
	elseif hierarchy_pos == 6
		return 3
	elseif hierarchy_pos == 8
		return 4
	elseif hierarchy_pos == 10
		return 5
	else
		error("Incorrect hierarchy position.")
		return -1
	end
end

function load_file_to_str_vector(file_path::String,file_name::String) :: Vector{String}
	v :: Vector{String} = []
        for line in eachline(file_path * file_name)
		push!(v,line)
	end
	return v
end

function save_matrix(matrix::Matrix{Float64},path::String,file_name::String) :: Nothing
	open(path * file_name, "w") do io
		writedlm(io, matrix, ',')
	end
	return nothing
end

function load_hs_codes(	td_hs_codes_database_path::String,
			td_hs_codes_database_file_name::String ) :: Vector{HsCode1}
	field_separator = "\t"
	hs_codes :: Vector{HsCode1} = []
    i = -1
    for line in eachline(td_hs_codes_database_path * td_hs_codes_database_file_name)
        i = i+1
        if i > 0
            segm_line = split(line,field_separator)
			symb = segm_line[2]
			desc_lvl_1 = segm_line[3]
			desc_lvl_2 = segm_line[4]
			desc_lvl_3 = segm_line[5]
			desc_lvl_4 = segm_line[6]
			desc_lvl_5 = segm_line[7]
			hsc = HsCode1(symb,desc_lvl_1,desc_lvl_2,desc_lvl_3,desc_lvl_4,desc_lvl_5)
			push!(hs_codes,hsc)
		end
	end
	return hs_codes
end

function training_set_files_from_hs_codes_1(	td_hs_codes_database_path::String,
					    td_hs_codes_database_file_name::String,
						training_set_path::String,
						training_set_dir::String,
						separator_file_name::String,
						verbosity::Int )
        path = training_set_path * training_set_dir
	refresh_dir(path)
	separator = create_separator()
	save_matrix(separator,path,separator_file_name)
	s = " "
	hs_codes = load_hs_codes(td_hs_codes_database_path,td_hs_codes_database_file_name)
	nhsc = lastindex(hs_codes)
	k = 0
	for i = 1:nhsc
		desc_a_lvl_1 = hs_codes[i].description_level_1
		desc_a_lvl_2 = hs_codes[i].description_level_2
		desc_a_lvl_3 = hs_codes[i].description_level_3
		desc_a_lvl_4 = hs_codes[i].description_level_4
		desc_a_lvl_5 = hs_codes[i].description_level_5
		text_a = desc_a_lvl_1 * s * desc_a_lvl_2 * s * desc_a_lvl_3 * s * desc_a_lvl_4 * s * desc_a_lvl_5
        	text_a_num = text_matrix(text_a)
		for j = 1:nhsc
			k += 1
			desc_b_lvl_1 = hs_codes[j].description_level_1
			desc_b_lvl_2 = hs_codes[j].description_level_2
			desc_b_lvl_3 = hs_codes[j].description_level_3
			desc_b_lvl_4 = hs_codes[j].description_level_4
			desc_b_lvl_5 = hs_codes[j].description_level_5
			text_b = desc_b_lvl_1 * s * desc_b_lvl_2 * s * desc_b_lvl_3 * s * desc_b_lvl_4 * s * desc_b_lvl_5
            		text_b_num = text_matrix(text_b)
            		input = vcat(text_a_num,separator,text_b_num)
			save_input(path,k,input,true)
			relatedness_num = 0.0
			if desc_a_lvl_1 == desc_b_lvl_1
				relatedness_num += 0.2
				if desc_a_lvl_2 == desc_b_lvl_2
					relatedness_num += 0.2
					if desc_a_lvl_3 == desc_b_lvl_3
						relatedness_num += 0.2
						if desc_a_lvl_4 == desc_b_lvl_4
							relatedness_num += 0.2
							if desc_a_lvl_5 == desc_b_lvl_5
								relatedness_num += 0.2
							end
						end
					end
				end
			end
			save_desired_output(path,k,[relatedness_num])
			display_pair_add_msg(verbosity,k)
		end
	end
	save_number_of_training_pairs(path,k)
end

function training_set_files_from_sick(	sick_database_path::String,
					training_set_path::String,
					training_set_dir::String,
					separator_file_name::String,
					verbosity::Int )
        path = training_set_path * training_set_dir
	refresh_dir(path)
	separator = create_separator()
	save_matrix(separator,path,separator_file_name)
        i = -1
        for line in eachline(sick_database_path * "SICK.txt")
            i = i+1
            if i > 0
                segm_line = split(line,"\t")
		text_a = String(segm_line[2])
		text_b = String(segm_line[3])
                text_a_num = text_matrix(text_a)
                text_b_num = text_matrix(text_b)
                input = vcat(text_a_num,separator,text_b_num)
		save_input(path,i,input,true)
                relatedness = segm_line[5]
                relatedness_num = parse(Float64,relatedness)
                relatedness_num = (relatedness_num - 1.0) / 4.0 # scale 1-5 normalized to 0-1
		save_desired_output(path,i,[relatedness_num])
		display_pair_add_msg(verbosity,i)
            end
        end
	save_number_of_training_pairs(path,i)
end

function save_input(path::String,input_id::Int64,input::Matrix{Float64},fixed_height::Bool)
	filepn = path * "input_" * string(input_id) * ".txt"
	open(filepn, "w") do io
		writedlm(io, input, ',')
	end
	if fixed_height
		(_,c) = size(input)
		line = "0.0"
		f = 1
		while f < c
			line *= ",0.0"
			f += 1
		end
		line *= "\n"
		io = open(filepn, "a")
		io2 = open(filepn, "r")
		n = 0
		for l in eachline(io2)
			n += 1
		end
		block = line
		n += 1
		while n < fixed_input_matrix_height
			block *= line
			n += 1
		end
		write(io,block)
		close(io)
	end
end

function save_desired_output(path::String,output_id::Int64,output::Vector{Float64})
	open(path * "desired_output_" * string(output_id) * ".txt", "w") do io
		writedlm(io, output, ',')
	end
end

function save_number_of_training_pairs(path::String,number_of_pairs::Int)
	io = open(path * "number_of_training_pairs.txt", "w")
	write(io,string(number_of_pairs))
	close(io)
end

function display_pair_add_msg(verbosity::Int,pair_id::Int)
	if verbosity > 1
        	print("Training pair ")
                print(pair_id)
                print(" added to the training set.\n")
        end
end

function load_training_set(training_set_path::String,training_set_dir::String,sparse_input::Bool)
        path = training_set_path * training_set_dir
		num_pairs = readline(path * "number_of_training_pairs.txt")
		n = parse(Int,num_pairs)
		training_set = Vector{Tuple{Matrix{Float64},Vector{Float64}}}(undef,n)
		for i = 1:n
			desired_output = readdlm(path * "desired_output_" * string(i) * ".txt", ',', Float64, '\n')
			desired_output = matrix_to_vector(desired_output)
			input = readdlm(path * "input_" * string(i) * ".txt", ',', Float64, '\n')
			if sparse_input
				input = sparse(input)
			end
			training_set[i] = (input,desired_output)
		end
		return training_set
end

end  # module Preprocessing
