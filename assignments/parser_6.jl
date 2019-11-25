



function parse_data_flatten(filename, skip_lines = []) # parse as a flatten vector using ind vector as an identifyer for the individual
    printstyled("parsing log: $filename\n", color = :green)
    f = open(filename, "r")
    lines = readlines(f)
    close(f)
    
    ind = Int[]
    y = Int[]
    subject = String[]
    kids = Int[] 
    row = 1
    for n in 2:length(lines)
        if n in skip_lines
            continue
        end
        line = lines[n]
        sline = split(line, "\t")
        push!(subject, sline[1])
        for sl in sline[4:end]
            if sl != ""
                time = parse(Int, sl)
                age = parse(Int,sline[3])
                kid = 1*(age<=12)
                push!(y, time)
                push!(ind, row) 
                push!(kids, kid) 

            end
        end
        row += 1
    end
    return y, ind, subject, kids
end

function parse_data_array(filename, skip_lines = []) # parse as a vector of vector
    printstyled("parsing log: $filename\n", color = :green)
    f = open(filename, "r")
    lines = readlines(f)
    close(f)
    
    y = Vector{Vector{Int}}()
    subject = String[]
    kids = Int[]
    for n in 2:length(lines)
        if n in skip_lines
            continue
        end
        line = lines[n]
        sline = split(line, "\t")
        push!(subject, sline[1])
        age = parse(Int,sline[3])
        kid = 1*(age<=12)        
        push!(kids, kid)
        y_j = Int[]
        for sl in sline[4:end]
            if sl != ""
                time = parse(Int, sl)
                push!(y_j, time)
            end
        end
        y = push!(y, y_j)
    end
    return y, subject, kids
end


if true # testing parsers
    using PyPlot
    close("all")

    # read data from csv
    filename = (@__DIR__) * raw"/data/ABDA 2019 -- Reaction time - Sheet1.tsv"

    skip_lines = [14,15,24,25,26,27,42] .- 1 


    y, ind, subject, kids = parse_data_flatten(filename, skip_lines)
    figure()
    for j in 1:ind[end]
        i = 1:sum(ind .== j)
        color = any(kids[ind .== j] .== 1) ? :red : :black 
        subplot(121)
        plot(i, y[ind .== j], ".-", alpha = .5, color = color)
        ylabel("reaction time (ms)")
        xlabel("attempt")
        xticks(1:2:20)
        subplot(122)
        plot(i, log.(y[ind .== j]), ".-", alpha = .5, color = color)
        ylabel("log(reaction time)")
        xlabel("attempt")
        xticks(1:2:20)

    end
    tight_layout()

    y, subject, kids = parse_data_array(filename, skip_lines)
    figure()
    for j in 1:length(y)
        i = 1:length(y[j])
        
        color = kids[j]==1 ? :red : :black
        plot(i, y[j], ".-", alpha = .5, color = color)
    end

end






