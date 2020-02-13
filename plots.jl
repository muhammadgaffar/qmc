using Plots
function Plots.plot(G0::T; mode=:both, kwargs...) where T <: GreenFunction
    p = plot(framestyle=:box)
    if mode == :both
        plot!(p,G0.w,real(G0.data), label = "real", kwargs...)
        plot!(p,G0.w,imag(G0.data), label = "imag", kwargs...)
    elseif mode == :real
        plot!(p,G0.w,real(G0.data), label = "", kwargs...)
    elseif mode == :imag
        plot!(p,G0.w,imag(G0.data), label = "", kwargs...)
    end
 end
