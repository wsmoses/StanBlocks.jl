using StanBlocks, PosteriorDB, BridgeStan, StanLogDensityProblems, LogDensityProblems, Logging, Test, Statistics, BenchmarkTools
import Enzyme, Zygote, Mooncake
import UnicodePlots


zygote(f, x) = Zygote.withgradient(f, x)
renzyme(f, x) = begin 
    g = zero(x)
    _, rv = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, Enzyme.Const(f), 
        Enzyme.Active, 
        Enzyme.Duplicated(x, g)
    )
    (rv, g)
end 
tryinferred(f, args...; handler=nothing) = try 
    @inferred f(args...)
    return true
catch e
    if hasproperty(e, :msg) && contains(e.msg, "does not match inferred return type")
        @info "$(typeof(f)): type inference failed, not continuing"
    else
        if isnothing(handler)
            @warn "$(typeof(f)): $e"
        else
            handler(e)
        end
    end
    return false
end
trybenchmark(x_f, f, args...) = try
    @benchmark $f($args..., x) setup=(x=$x_f())
catch e
    @warn "$(typeof(f)): $e"
    nothing
end

test_lpdf(posterior_name; n_draws=10) = begin 
    post = PosteriorDB.posterior(pdb, posterior_name)
    jlpdf = StanBlocks.julia_implementation(post)
    ismissing(jlpdf) && return 
    prob = with_logger(ConsoleLogger(stderr, Logging.Error)) do 
        StanProblem(
            PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(post), "stan")), 
            PosteriorDB.load(PosteriorDB.dataset(post), String)
        )
    end
    n = LogDensityProblems.dimension(prob)
    tryinferred(jlpdf, randn(n)) || return
    slpdf = Base.Fix1(LogDensityProblems.logdensity, prob)
    slpdfs = zeros(n_draws)
    jlpdfs = zeros(n_draws)
    for i in 1:n_draws
        x = randn(n)
        slpdfs[i] = slpdf(x)
        jlpdfs[i] = jlpdf(x)
    end
    jlpdfs .+= mean(slpdfs - jlpdfs)
    iscorrect = isapprox(slpdfs, jlpdfs)
    printstyled("$posterior_name\n"; color = iscorrect ? :blue : :red)
    iscorrect || return
    xf() = randn(n)
    rule = Mooncake.build_rrule(jlpdf, randn(n))
    trials = [
        trybenchmark(xf, LogDensityProblems.logdensity, prob)
        trybenchmark(xf, jlpdf)
        trybenchmark(xf, LogDensityProblems.logdensity_and_gradient, prob)
        trybenchmark(xf, zygote, jlpdf)
        trybenchmark(xf, renzyme, jlpdf)
        trybenchmark(xf, Mooncake.value_and_gradient!!, rule, jlpdf)
    ]
    mask =  .!isnothing.(trials)
    labels = ["Primal Stan", "Primal Julia", "Gradient Stan", "Zygote", "Enzyme", "Mooncake"][mask]
    runtimes = map(t->mean(t).time, trials[mask])
    display(UnicodePlots.barplot(
        labels,
        runtimes ./ minimum(runtimes),
        title="Mean runtime",
        xscale=:log10
    ))
    # for g in (zygote, renzyme)
    #     println("Gradient Stan vs $g:")
    #     tryinferred(g, jlpdf, randn(n)) || continue
    #     jtrial = @benchmark $g($jlpdf, x) setup=(x=randn($n))
    #     display(judge(mean(jtrial), mean(strial)))
    # end
    # println("Gradient Stan vs Mooncake:")
    # display(judge(mean(jtrial), mean(strial)))
end


pdb = PosteriorDB.database()
posterior_names = PosteriorDB.posterior_names(pdb)
order = sortperm(posterior_names; by=posterior_name->length(PosteriorDB.load(PosteriorDB.implementation(PosteriorDB.model(PosteriorDB.posterior(pdb, posterior_name)), "stan"))))
posterior_names = posterior_names[order]
map(test_lpdf, posterior_names)

