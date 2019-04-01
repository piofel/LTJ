module MathematicalFunctions

function random_vector(vec_length,lower_bound,upper_bound)
        ll = lower_bound * ones(vec_length)
        rand(vec_length) * (upper_bound - lower_bound) + ll
end

function random_array(num_rows,num_columns,lower_bound,upper_bound)
        ll = lower_bound * ones(num_rows,num_columns)
        rand(num_rows,num_columns) * (upper_bound - lower_bound) + ll
end

# Rectifier (ramp) function
function rectifier(x)
        if x > 0.0
                return x
        else
                return 0.0
        end
end

# Hyperbolic tangent derivative
tanh_derivative(x) = sech(x)^2

# Sigmoid function
sigmoid(x) = 1.0 / (1 + exp(-x))

# The first derivative of the sigmoid function
sigmoid_derivative(x) = sigmoid(x) * (1.0 - sigmoid(x))

# Binary threshold function
function heaviside_function(x)
        if x >= 0.0
                1.0
        else
                0.0
        end
end

# Symmetrical binary threshold function
function sym_binary(x)
        if x >= 0.0
                1.0
        else
                -1.0
        end
end

# Identity function
id(x) = x

# Identity function derivative
id_derivative(x) = 1.0

function convolution(input_matrix::Matrix{Float64},filter_matrix::Matrix{Float64})
    (imh,imw) = size(input_matrix)
    (fh,fw) = size(filter_matrix)
    rh = imh-fh+1
    rw = imw-fw+1
    result = Array{Float64,2}(undef,rh,rw)
    if fh <= imh
	if fw <= imw
    	    for i = 1:rh
	    	for j = 1:rw
		    hp = input_matrix[i:i+fh-1,j:j+fw-1] .* filter_matrix  # Hadamard-Schur (element-wise) product
		    result[i,j] = sum(hp)
	    	end
	    end
    	else
		error("Intput matrix width less than filter width.")
    	end
    else
	error("Intput matrix height less than filter height.")
    end
    return result
end

function full_convolution(input_matrix::Matrix{Float64},filter_matrix::Matrix{Float64})
    (imh,imw) = size(input_matrix)
    (fh,fw) = size(filter_matrix)
	zv = zeros(fh-1,imw)
	zvi = vcat(zv,input_matrix)
	zvizv = vcat(zvi,zv)
	(zvizvh,_) = size(zvizv)
	zh = zeros(zvizvh,fw-1)
	zhzvizv = hcat(zh,zvizv)
	zhzvizvzh = hcat(zhzvizv,zh)
	return convolution(zhzvizvzh,filter_matrix)
end

# Creates a matrix with vector's elements on the main diagonal
function diag(vector::Array{Float64,1})
	n = lastindex(vector)
	m = zeros(n,n)
	for i = 1:n
		m[i,i] = vector[i]
	end
	return m
end

function maximum_fun_derivative(input::Matrix{Float64}) :: Matrix{Float64}
	max = maximum(input)
	(r,c) = size(input)
	d = Matrix{Float64}(undef,r,c)
	for i = 1:r
		for j = 1:c
			if input[i,j] == max
				d[i,j] = 1.0
			else
				d[i,j] = 0.0
			end
		end
	end
	return d
end

function matrix_to_vector(m::Matrix{Float64})
	(r,c) = size(m)
	if c == 1
		v = reshape(m,r)
		return v
	else
		return m
	end
end

end  # module MathematicalFunctions
