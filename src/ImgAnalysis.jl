module ImgAnalysis

# = = = = = = = = = = = = = = = = = = = = = #
# Last-In, First-Out Stack for DFS Indexing  #
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

end # module ImgAnalysis
