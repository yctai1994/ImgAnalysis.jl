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

const NEIGHBOR_ROWOPS = ( 0, -1,  0, +1)
const NEIGHBOR_COLOPS = (-1,  0, +1,  0)

function count_area(img::BitMatrix)
    m, n = size(img)

    v = BitArray(false for i in 1:m, j in 1:n)
    r = Vector{NTuple{3,Int}}(undef, 0)
    s = IndexStack((m - 1) * n >> 1)
    a = 0

    for j in axes(v, 2), i in axes(v, 1)
        if @inbounds !v[i,j]
            @inbounds v[i,j] = true
            if @inbounds img[i,j]
                push!(s, i, j)
                a += 1
            
                while true
                    length(s) == 0 && break
                    ri, ci = pop!(s)
                    for k in 1:4
                        @inbounds ki = ri + NEIGHBOR_ROWOPS[k]
                        @inbounds kj = ci + NEIGHBOR_COLOPS[k]
                        if 0 < ki ≤ m && 0 < kj ≤ n && @inbounds !v[ki, kj]
                            @inbounds v[ki, kj] = true
                            if @inbounds img[ki, kj]
                                push!(s, ki, kj)
                                a += 1
                            end
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
