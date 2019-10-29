using ABDA
using PyPlot
using Random

close("all")


# read data from csv
filename = (@__DIR__)*raw"/data/ABDA 2019 -- Reaction time - Sheet1.tsv"

skip_lines = [14,15,24,25,26,27,42] .- 1 

function parse_log_flatten(filename, skip_lines = []) # parse as a flatten vector using ind vector as an identifyer for the individual
    printstyled("parsing log: $filename\n", color = :green)
    f = open(filename,"r")
    lines = readlines(f)
    close(f)
    
    ind = Int[]
    y = Int[]
    subject = String[]
    row = 1
    for n in 2:length(lines)
        if n in skip_lines
            continue
        end
        line = lines[n]
        sline = split(line,"\t")
        push!(subject,sline[1])
        for sl in sline[4:end]
            if sl!=""
                time = parse(Int,sl)
                push!(y,time)
                push!(ind,row) 

            end
        end
        row += 1
    end
    return y, ind, subject
end

function parse_log_array(filename, skip_lines=[]) # parse as a vector of vector
    printstyled("parsing log: $filename\n", color = :green)
    f = open(filename,"r")
    lines = readlines(f)
    close(f)
    
    y = Vector{Vector{Int}}()
    for n in 2:length(lines)
        if n in skip_lines
            continue
        end
        line = lines[n]
        sline = split(line,"\t")
        y_j = Int[]
        for sl in sline[4:end]
            if sl!=""
                time = parse(Int,sl)
                push!(y_j,time)
            end
        end
        y = push!(y,y_j)
    end
    return y, subject
end


y, ind, subject = parse_log_flatten(filename, skip_lines)
figure()
for j in 1:ind[end]
    i = 1:sum(ind .== j)
    subplot(121)
    plot(i, y[ind .== j],".-", alpha=.5)
    ylabel("reaction time (ms)")
    xlabel("attempt")
    xticks(1:2:20)
    subplot(122)
    plot(i, log.(y[ind .== j]),".-", alpha=.5)
    ylabel("log(reaction time)")
    xlabel("attempt")
    xticks(1:2:20)

end
tight_layout()


# Y, subject2 = parse_log_array(filename)
# figure()
# for j in 1:length(Y)
#     i = 1:length(Y[j])
#     plot(i, Y[j],".-", alpha=.5)
# end


