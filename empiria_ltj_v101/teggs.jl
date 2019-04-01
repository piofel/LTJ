module Teggs

using Main.ConvNetMultilayerPerceptronHybrid: ConvPercHybrid
using Main.LanguageTechnology: compare_meaning
using Main.Preprocessing: HsCode1, load_hs_codes, load_file_to_str_vector, text_matrix

using Gtk: GtkWindow, GtkGrid, GtkButton, GtkLabel, GtkEntry, showall, set_gtk_property!, signal_connect, get_gtk_property

function find_commodity(	software_name::String,
				software_version::String,
				result_display_treshold::Float64,
				net::ConvPercHybrid,
				hs_codes_database_path::String,
				hs_codes_database_file_name::String,
				training_set_path::String,
				training_set_dir::String,
				separator_file_name::String )
	s = 10
	m = 10
	w = 80
	win = GtkWindow(software_name * " " * software_version * " for TEGGS")
	set_gtk_property!(win,:resizable,false)
	g = GtkGrid()
	set_gtk_property!(g,:column_spacing,s)
	set_gtk_property!(g,:row_spacing,s)
	set_gtk_property!(g,:margin_top,m)
	set_gtk_property!(g,:margin_bottom,m)
	set_gtk_property!(g,:margin_left,m)
	set_gtk_property!(g,:margin_right,m)
	e1 = GtkEntry()
	e2 = GtkEntry()
	l1 = GtkLabel("")
	l2 = GtkLabel("")
	l3 = GtkLabel("Not mandatory: a known HS code (or its part) of some country:")
	sb = GtkButton("Search")
	crb = GtkButton("Clear results")
	set_gtk_property!(e1,:width_chars,w)
	set_gtk_property!(e2,:text,"")
	set_gtk_property!(l1,:selectable,true)
	set_gtk_property!(l1,:wrap,true)
	set_gtk_property!(l1,:max_width_chars,w)
	g[1,1] = GtkLabel("Please enter the commodity description:")
	g[1:3,2] = e1
	g[1,3] = l3
	g[2:3,3] = e2
	g[1,4] = sb
	g[2:3,4] = crb
	g[1:3,5] = l2
	g[1:3,6] = l1
	data = (result_display_treshold,e1,e2,l1,l2,net,hs_codes_database_path,hs_codes_database_file_name,training_set_path,training_set_dir,separator_file_name)
	signal_connect(search_button_clicked_callback, sb, "clicked",Nothing,(),false,data)
	signal_connect(clear_result_button_clicked_callback,crb,"clicked",Nothing,(),false,(l1,l2))
	push!(win,g)
	showall(win)
end

function search_button_clicked_callback(button,data)
		length_of_hs_codes_common_part :: UInt8 = 6
		(treshhold,e1,e2,l1,l2,net,hs_codes_database_path,hs_codes_database_file_name,training_set_path,training_set_dir,separator_file_name) = data
		hs_codes = load_hs_codes(hs_codes_database_path,hs_codes_database_file_name)
		nhsc = lastindex(hs_codes)
		knhs = get_gtk_property(e2,:text,String)
		if knhs != ""
			ncknhs = lastindex(knhs)
			hs_codes_2 :: Vector{HsCode1} = []
			if ncknhs > length_of_hs_codes_common_part
				knhs = knhs[1:length_of_hs_codes_common_part]
				ncknhs = lastindex(knhs)
			end
			for i = 1:nhsc
				hsc = hs_codes[i]
				if hsc.symbol[1:ncknhs] == knhs
					push!(hs_codes_2,hsc)
				end
			end
			hs_codes = hs_codes_2
			nhsc = lastindex(hs_codes)
		end
		text = get_gtk_property(e1,:text,String)
		text_mat = text_matrix(text)
		s = " "
		eval_hs_codes :: Vector{Tuple{HsCode1,Float64}} = []
		for i = 1:nhsc
			hsc = hs_codes[i]
			desc = hsc.description_level_1 * s * hsc.description_level_2 * s * hsc.description_level_3 * s * hsc.description_level_4 * s * hsc.description_level_5
			r = compare_meaning(text_mat,desc,net,training_set_path,training_set_dir,separator_file_name)
			if r >= treshhold
				push!(eval_hs_codes,(hsc,r))
			end
		end
		sorted_hs_codes = sort_proposed_hs_codes(eval_hs_codes)
		ns = lastindex(sorted_hs_codes)
		msg = ""
		if ns > 0
			msg = "\u2116\tHS code\t\t\tRelated\tDescription\n"
			msg *= "--\t---------\t\t\t--------\t------------\n"
		else
			set_gtk_property!(l2,:label,"")
			msg = "No results."
		end
		if ns > 1
			set_gtk_property!(l2,:label,"There is more than one result; please provide more details for disambiguation.")
		else
			set_gtk_property!(l2,:label,"")
		end
		for i = 1:ns
			(hsc,r) = sorted_hs_codes[i]
			r = round(r*100,digits=1)
			desc = hsc.description_level_1 * s * hsc.description_level_2 * s * hsc.description_level_3 * s * hsc.description_level_4 * s * hsc.description_level_5
			msg *= string(i) * ".) \t" * hsc.symbol * "\t" * string(r) * "%\t" * desc * "\n"
		end
		set_gtk_property!(l1,:label,msg)
		return nothing
end

function sort_proposed_hs_codes(hs_codes::Vector{Tuple{HsCode1,Float64}})
	n = lastindex(hs_codes)
	sorted_hs_codes = Vector{Tuple{HsCode1,Float64}}(undef,n)
	for h = 1:n
		max = 0.0
		n = lastindex(hs_codes)
		maxid :: Int64 = 1
		for i = 1:n
			(hsc,r) = hs_codes[i]
			if r > max
				max = r
				maxid = i
			end
		end
		sorted_hs_codes[h] = hs_codes[maxid]
		deleteat!(hs_codes,maxid)
	end
	return sorted_hs_codes
end

function clear_result_button_clicked_callback(button,data)
	(l1,l2) = data
	set_gtk_property!(l1,:label,"")
	set_gtk_property!(l2,:label,"")
	return nothing
end

end  # module Teggs
