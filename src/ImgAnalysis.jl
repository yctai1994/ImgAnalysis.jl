module ImgAnalysis

import FixedPointNumbers: N0f8
import Colors: RGB
import LinearAlgebra: BLAS

const VecI  = AbstractVector
const VecO  = AbstractVector
const MatI  = AbstractMatrix
const MatO  = AbstractMatrix

# = = = = = = = = = = = = = = = = = = = = = #
# RGB to Gray Scale Conversion              #
# = = = = = = = = = = = = = = = = = = = = = #

@inline rgb_to_gray(rgb::RGB{N0f8}) = 0.298N0f8 * rgb.r + 0.588N0f8 * rgb.g + 0.114N0f8 * rgb.b
@inline rgb_to_gray(img::MatI{RGB{N0f8}}) = rgb_to_gray!(similar(img, Float32), img)

function rgb_to_gray!(des::MatO{Float32}, src::MatI{RGB{N0f8}})
    @simd for i in eachindex(des)
        @inbounds des[i] = Float32(rgb_to_gray(src[i]))
    end
    return des
end

# = = = = = = = = = = = = = = = = = = = = = #
# Polynomial Shading Correction             #
# = = = = = = = = = = = = = = = = = = = = = #

BLAS.set_num_threads(4)

function dot(x::VecI{Tx}, y::VecI{Ty}, n::Int) where {Tx<:Real,Ty<:Real}
    r = 0.0
    m = mod(n, 5)
    if m ≠ 0
        for i in 1:m
            @inbounds r += x[i] * y[i]
        end
        n < 5 && return r
    end
    m += 1
    for i in m:5:n
        @inbounds r += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + x[i+3] * y[i+3] + x[i+4] * y[i+4]
    end
    return r
end

dot(x::VecI{Tx}, n::Int) where Tx<:Real = dot(x, x, n)

function dot(x::VecI{Tx}, m::Int, A::MatI{TA}, y::VecI{Ty}, n::Int) where {Tx<:Real, TA<:Real, Ty<:Real}
    ret = 0.0
    for j in eachindex(1:n)
        @inbounds yj = y[j]
        if !iszero(yj)
            tmp = 0.0
            for i in eachindex(1:m)
                @inbounds tmp += A[i,j] * x[i]
            end
            ret += tmp * yj
        end
    end
    return ret
end

function legendre!(p::VecO, x::Real, n::Int)
    @inbounds p[1] = 1.0
    if n > 1
        @inbounds p[2] = x
        for ℓ in 3:n+1
            @inbounds p[ℓ] = ((2 * ℓ - 3) * x * p[ℓ-1] - (ℓ - 2) * p[ℓ-2]) / (ℓ - 1)
        end
    end
    return p
end

function legendre2D!(des::VecO, pK::VecI, ΩK::VecI, pL::VecI, ΩL::VecI)
    K = length(pK)
    for ℓ in eachindex(pL)
        pad = (ℓ - 1) * K
        @inbounds qLℓ = pL[ℓ] / ΩL[ℓ]
        @simd for k in eachindex(pK)
            @inbounds des[k + pad] = qLℓ * pK[k] / ΩK[k]
        end
    end
end

struct Corrector
    imgI::Matrix{Float64}
    imgO::Matrix{Float64}
    PKx::Matrix{Float64}
    PLy::Matrix{Float64}
    ΩKx::Vector{Float64}
    ΩLy::Vector{Float64}
    ΦKL::Matrix{Float64}
    aKL::Matrix{Float64}
    K::Int
    L::Int

    function Corrector(img::MatI{Float32}, K::Int, L::Int)
        m, n = size(img)
        one2m = axes(img, 1)
        one2n = axes(img, 2)

        imgI = Matrix{Float64}(undef, m, n)
        imgO = Matrix{Float64}(undef, m, n)
        @simd for i in eachindex(img)
            @inbounds imgI[i] = img[i]
        end

        Kp1 = K + 1
        Lp1 = L + 1
        PKx = Matrix{Float64}(undef, Kp1, m)
        PLy = Matrix{Float64}(undef, Lp1, n)

        ai = 2 // (m - 1)
        aj = 2 // (n - 1)
        bi = (m + 1) // (m - 1)
        bj = (n + 1) // (n - 1)

        for i in one2m
            legendre!(view(PKx, :, i), ai * i - bi, K)
        end

        for j in one2n
            legendre!(view(PLy, :, j), aj * j - bj, L)
        end

        ΩKx = Vector{Float64}(undef, Kp1)
        ΩLy = Vector{Float64}(undef, Lp1)

        for k in eachindex(ΩKx)
            @inbounds ΩKx[k] = dot(view(PKx, k, :), m)
        end

        for ℓ in eachindex(ΩLy)
            @inbounds ΩLy[ℓ] = dot(view(PLy, ℓ, :), n)
        end

        ΦKL = Matrix{Float64}(undef, Kp1 * Lp1, m * n)

        for j in one2n, i in one2m
            legendre2D!(view(ΦKL, :, i + (j - 1) * m), view(PKx, :, i), ΩKx, view(PLy, :, j), ΩLy)
        end

        aKL = Matrix{Float64}(undef, Kp1, Lp1)

        BLAS.gemv!('N', 1.0, ΦKL, view(imgI, :), 0.0, view(aKL, :))

        return new(imgI, imgO, PKx, PLy, ΩKx, ΩLy, ΦKL, aKL, K, L)
    end
end

function correction(c::Corrector)
    imgI, imgO = c.imgI, c.imgO
    Kp1 = c.K + 1
    Lp1 = c.L + 1
    for j in axes(imgI, 2), i in axes(imgI, 1)
        @inbounds imgO[i,j] = imgI[i,j] / dot(view(c.PKx, :, i), Kp1, c.aKL, view(c.PLy, :, j), Lp1)
    end
    return imgO
end

# = = = = = = = = = = = = = = = = = = = = = #
# Background Subtraction by Data Leveling   #
# = = = = = = = = = = = = = = = = = = = = = #

function leveling(img::MatI{T}) where T<:Real
    m, n = size(img)
    return leveling!(Matrix{T}(undef, m, n), img, m, n)
end

function leveling!(des::MatO{T}, src::MatI{T}, m::Int, n::Int) where T<:Real
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

# = = = = = = = = = = = = = = = = = = = = = #
# Kernel K-Means Clustering                 #
# = = = = = = = = = = = = = = = = = = = = = #

#=
Kernel function:
----------------
    exp(-‖Δh‖/γH - ‖Δw‖/γW - ‖Δg‖/γG)

Parameters:
----------------
    1. Δh, γH := height diff. and height factor (row-indexing)
    2. Δw, γW := width diff. and width factor (column-indexing)
    3. Δg, γG := grayscale diff. and grayscale factor
=#
@inline kernel(Δh::Real, Δw::Real, Δg::Real, γH::Real, γW::Real, γG::Real) = exp(-abs(Δh) / γH - abs(Δw) / γW - abs(Δg) / γG)

function kernel!(KNN::MatI, XMN::MatI; γH::Real=3.0, γW::Real=3.0, γG::Real=4e-2)
    N = size(XMN, 2)
    for j in 1:N, i in j:N
        @inbounds KNN[i,j] = kernel(XMN[1,i] - XMN[1,j], XMN[2,i] - XMN[2,j], XMN[3,i] - XMN[3,j], γH, γW, γG)
    end
    return KNN
end

#=
K-means++ Initialization:
-------------------------
    For all points 𝐱, the distance 𝐷(𝐱) is defined as
        𝐷(𝐱) = (distance to the nearest centroid)

    The new centroid is chosen by
        (new centroid) = 𝗮𝗿𝗴𝗺𝗮𝘅 𝐷(𝐱) ∀ 𝐱

Parameters:
-------------------------
    1. WNK ∈ ℝ(N × K) := Matrix of weights
    2. DNK ∈ ℝ(N × K) := Matrix of point-to-centroid distances
    3. KNN ∈ ℝ(N × N) := Kernel (Gram) matrix
=#
function kmeanspp!(WNK::MatIO, DNK::MatI, KNN::MatI)
    N, K = size(WNK)
    m = rand(1:N) # randomly choose a point as the 1st centroid
    k = 1         # counting of found centroids
    @simd for n in axes(DNK, 1)
        @inbounds DNK[n,k] = KNN[n,n] + KNN[m,m] - 2.0 * KNN[max(m,n), min(m,n)]
    end

    while k < K
        x2m_max = -Inf
        m = 0
        for n in axes(DNK, 1)
            x2m_min = minimum(view(DNK, n, 1:k))
            if x2m_min > x2m_max
                x2m_max = x2m_min
                m = n
            end
        end
        k += 1
        @simd for n in axes(DNK, 1)
            @inbounds DNK[n,k] = KNN[n,n] + KNN[m,m] - 2.0 * KNN[max(m,n), min(m,n)]
        end
    end

    for n in axes(WNK, 1)
        ktarget = argmax(view(DNK, n, :))
        @simd for k in axes(WNK, 2)
            @inbounds WNK[n,k] = ifelse(k ≡ ktarget, 1.0, 0.0)
        end
    end
    return WNK
end

function update!(WNK::MatI, DNK::MatIO, N1K::MatI, KNN::MatI)
    sum!(N1K, WNK)

    N = size(WNK, 1)
    for k in axes(WNK, 2)
        N1k = @inbounds N1K[k]
        WNk = view(WNK, :, k)
        tmp = dot(WNk, N, KNN, WNk, N) / (N1k * N1k)
        @simd for n in axes(WNK, 1)
            @inbounds DNK[n,k] = tmp
        end

        BLAS.symv!('L', -2.0 / N1k, KNN, WNk, 1.0, view(DNK, :, k))
    end

    change = 0
    for n in axes(WNK, 1)
        kOld = findfirst(isone, view(WNK, n, :))
        kNew = argmin(view(DNK, n, :))
        if kOld ≠ kNew
            @inbounds WNK[n,kOld], WNK[n,kNew] = 0.0, 1.0
            change += 1
        end
    end

    return change
end

function iterate!(WNK::MatI, DNK::MatI, N1K::MatI, KNN::MatI)
    changes = 1
    trapped = 0
    itcount = 0
    while changes ≠ 0 && trapped < 5 && itcount < 200
        change_ = update!(WNK, DNK, N1K, KNN)
        itcount += 1
        println("Current change = $change_ ($itcount)")
        if change_ ≠ changes
            changes = change_
            trapped = 0
        else
            trapped += 1
        end
    end
    return nothing
end

end # module ImgAnalysis
