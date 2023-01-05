import Plots, DelimitedFiles

function demo(
        src::AbstractMatrix{T}, res::AbstractMatrix{Int};
        ifsave::Bool=false, fname::String=""
    ) where T<:Real
    fig = Plots.plot(
        Plots.heatmap(src; aspect_ratio=:equal, yflip=true),
        Plots.heatmap(res; aspect_ratio=:equal, yflip=true);
        layout=(1, 2), size=(1200, 600)
    )
    ifsave && fname ≠ "" && Plots.savefig(fig, "$fname.png")
    return fig
end

save_result(fname::String, src::AbstractMatrix{Int};
            K::Int=-1, P::Real=-Inf, γH::Real=0.0, γW::Real=0.0, γG::Real=0.0) =
    DelimitedFiles.writedlm("$fname,K=$K,P=$P,γH=$γH,γW=$γW,γG=$γG.txt", src)

