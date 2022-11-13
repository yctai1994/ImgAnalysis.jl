module ImgAnalysis

# = = = = = = = = = = = = = = = = = = = = = #
# Last-In, First-Out Stack for DFS Indexing #
# = = = = = = = = = = = = = = = = = = = = = #
mutable struct IndexStack
    capacity::Int     # size of storage capacity
    rind::Vector{Int} # row-wise indices
    cind::Vector{Int} # column-wise indices
    sz::Int           # size of current storage
    ix::Int           # index of last-in

    function IndexStack(capacity::Int)
        rind = Vector{Int}(undef, capacity)
        cind = Vector{Int}(undef, capacity)
        return new(capacity, rind, cind, 0, 0)
    end
end

@inline Base.length(s::IndexStack) = s.sz

@inline function Base.push!(s::IndexStack, irow::Int, icol::Int)
    @boundscheck s.sz < s.capacity || throw(BoundsError())
    s.sz += 1
    tmp = s.ix + 1
    @inbounds s.rind[tmp] = irow
    @inbounds s.cind[tmp] = icol
    s.ix = tmp
    return s
end

@inline function Base.pop!(s::IndexStack)
    @boundscheck s.sz > 0 || throw(BoundsError())
    ix = s.ix
    irow = @inbounds s.rind[ix]
    icol = @inbounds s.cind[ix]
    s.sz -= 1
    s.ix = ix - 1
    return irow, icol
end

# = = = = = = = = = = = = = = = = = = = = = #
# Count Multiple Area in One Image by DFS   #
# = = = = = = = = = = = = = = = = = = = = = #

function count_area(img::BitMatrix)
    m, n = size(img)

    v = BitArray(false for i in 1:m, j in 1:n) # visited
    r = Vector{NTuple{3,Int}}(undef, 0)        # records
    s = IndexStack((m - 1) * n >> 1)           # stack
    a = 0                                      # area

    for j in axes(v, 2), i in axes(v, 1)
        if @inbounds !v[i,j]
            @inbounds v[i,j] = true
            if @inbounds img[i,j]
                push!(s, i, j)
                a += 1
            
                while true
                    length(s) == 0 && break
                    ix, jx = pop!(s)

                    jj = jx - 1
                    if 0 < jj && @inbounds !v[ix, jj]
                        @inbounds v[ix, jj] = true
                        if @inbounds img[ix, jj]
                            push!(s, ix, jj)
                            a += 1
                        end
                    end

                    ii = ix - 1
                    if 0 < ii && @inbounds !v[ii, jx]
                        @inbounds v[ii, jx] = true
                        if @inbounds img[ii, jx]
                            push!(s, ii, jx)
                            a += 1
                        end
                    end

                    jj = jx + 1
                    if jj ≤ n && @inbounds !v[ix, jj]
                        @inbounds v[ix, jj] = true
                        if @inbounds img[ix, jj]
                            push!(s, ix, jj)
                            a += 1
                        end
                    end

                    ii = ix + 1
                    if ii ≤ m && @inbounds !v[ii, jx]
                        @inbounds v[ii, jx] = true
                        if @inbounds img[ii, jx]
                            push!(s, ii, jx)
                            a += 1
                        end
                    end
                end

                push!(r, (i, j, a))
                a = 0
            end
        end
    end

    return r
end

end # module ImgAnalysis
