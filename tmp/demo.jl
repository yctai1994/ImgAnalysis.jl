import CairoMakie, DelimitedFiles

function demo(
        src::AbstractMatrix{T}, res::AbstractMatrix{Int};
        ifsave::Bool=false, fname::String=""
    ) where T<:Real
    nrow, ncol = size(res)
    ncat = maximum(res)
    fig = CairoMakie.Figure(resolution=(7 * 180, 6 * 180))
    ax1 = CairoMakie.Axis(fig[1,1], aspect=CairoMakie.DataAspect())
    hm1 = CairoMakie.heatmap!(ax1, 0:nrow, 0:ncol, src; colormap=(:grays, 1.0))
    ct1 = CairoMakie.contourf!(ax1, res; linewidth=0.0, colormap=(:Spectral_7, 0.4), levels=1:ncat+1)
    ct2 = CairoMakie.contour!(ax1, res; linewidth=10.0, colormap=(:Spectral_7, 1.0), levels=1:ncat+1)
    CairoMakie.Colorbar(fig[1,2], hm1)
    CairoMakie.Colorbar(fig[1,3], ct2)
    ifsave && fname ≠ "" && CairoMakie.save("$fname.png", fig)
    return fig
end

save_result(fname::String, src::AbstractMatrix{Int};
            K::Int=-1, P::Real=-Inf, γH::Real=0.0, γW::Real=0.0, γG::Real=0.0) =
    DelimitedFiles.writedlm("$fname,K=$K,P=$P,γH=$γH,γW=$γW,γG=$γG.txt", src)

