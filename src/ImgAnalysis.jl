module ImgAnalysis

import FixedPointNumbers: N0f8
import Colors: RGB

# = = = = = = = = = = = = = = = = = = = = = #
# Background Subtraction by Data Levelling  #
# = = = = = = = = = = = = = = = = = = = = = #

@inline rgb_to_gray(rgb::RGB{N0f8}) = 0.298N0f8 * rgb.r + 0.588N0f8 * rgb.g + 0.114N0f8 * rgb.b
@inline rgb_to_gray(img::Matrix{RGB{N0f8}}) = rgb_to_gray!(similar(img, Float32), img)

function rgb_to_gray!(des::Matrix{Float32}, src::Matrix{RGB{N0f8}})
    @simd for i in eachindex(des)
        @inbounds des[i] = Float32(rgb_to_gray(src[i]))
    end
    return des
end

function leveling(img::Matrix{T}) where T<:Real
    m, n = size(img)
    return leveling!(Matrix{T}(undef, m, n), img, m, n)
end

function leveling!(des::Matrix{T}, src::Matrix{T}, m::Int, n::Int) where T<:Real
    a1 = m - 1
    a2 = n - 1
    a3 = @inbounds src[m,n] - src[1,1]
    b1 = 1 - m
    b2 = n - 1
    b3 = @inbounds src[1,n] - src[m,1]

    nx = a2 * b3 - a3 * b2
    ny = a3 * b1 - a1 * b3
    nz = a1 * b2 - a2 * b1

    ip = argmin(src)

    d0 = @inbounds nx * ip[1] + ny * ip[2] + nz * src[ip]
    nx = nx / nz
    ny = ny / nz
    d0 = d0 / nz

    for j in axes(des, 2)
        @simd for i in axes(des, 1)
            @inbounds des[i,j] = src[i,j] + nx * i + ny * j - d0
        end
    end
    return des
end

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
