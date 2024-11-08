@inline log1m(x) = log(1-x)
ternary(c,t,f) = c ? t : f
# https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html#poisson-distribution-log-parameterization
@inline poisson_log_lpdf(n, alpha) = sum(@broadcasted(n * alpha - exp(alpha)))
# https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html#binomial-distribution-logit-parameterization
@inline binomial_logit_lpmf(args...) = binomial_logit_lpdf(args...)
@inline binomial_logit_lpdf(n, N, alpha) = sum(@broadcasted(n * loglogistic(alpha) + (N - n) * log1mlogistic(alpha)))
# https://mc-stan.org/docs/2_21/functions-reference/binomial-distribution.html
@inline binomial_lpmf(args...) = binomial_lpdf(args...)
@inline binomial_lpdf(n, N, theta) = sum(@broadcasted(n * log(theta) + (N - n) * log1m(theta)))
# https://mc-stan.org/docs/2_21/functions-reference/bernoulli-distribution.html
@inline bernoulli_lpmf(args...) = bernoulli_lpdf(args...)
@inline bernoulli_lpdf(args...) = sum(bernoulli_lpdf.(args...))
@inline bernoulli_lpdf(y::Real, theta::Real) = y == 1 ? log(theta) : log1m(theta)
@inline bernoulli_logit_lpmf(args...) = bernoulli_logit_lpdf(args...)
@inline bernoulli_logit_lpdf(y, alpha) = sum(@broadcasted (bernoulli_logit_lpdf(y, alpha)))
@inline bernoulli_logit_lpdf(y::Real, alpha::Real) = y == 1 ? loglogistic(alpha) : log1mlogistic(alpha)
@inline bernoulli_logit_glm_lpdf(y, X, alpha, beta) = bernoulli_logit_lpdf(y, alpha .+ X * beta)
# @inline bernoulli_logit_glm_lpdf(y, X::AbstractVector, alpha, beta) = bernoulli_logit_lpdf(y, alpha .+ X * beta)
# https://mc-stan.org/docs/2_21/functions-reference/beta-distribution.html
@inline beta_lpmf(args...) = beta_lpdf(args...)
@inline beta_lpdf(theta, alpha, beta) = sum(@broadcasted((alpha-1)*log(theta) + log1m(theta)*(beta-1)))
# https://mc-stan.org/docs/2_21/functions-reference/dirichlet-distribution.html
@inline dirichlet_lpdf(theta, alpha) = sum(@broadcasted(log(theta) * (alpha-1)))
# https://mc-stan.org/docs/2_21/functions-reference/gamma-distribution.html
# @ inline gamma_lpdf()
# https://mc-stan.org/docs/functions-reference/matrix_operations.html#exponentiated-quadratic-kernel
@inline gp_exp_quad_cov(x, sigma, length_scale) = @.(sigma^2 * exp(- .5 * square((x - x')/length_scale)))


@inline std_normal_lpdf(x) = -.5 * sum(@broadcasted(square(x)))
@inline normal_lpdf(x, mu, sigma) = begin
    bc = @broadcasted(square((x-mu)/sigma))
    -sum(@broadcasted(log(sigma))) * length(bc)/length(sigma) - .5 * sum(bc)
end
# https://mc-stan.org/docs/2_21/functions-reference/normal-id-glm.html
@inline normal_id_glm_lpdf(y,X,alpha,beta,sigma) = normal_lpdf(y, alpha .+ X * beta, sigma)
@inline lognormal_lpdf(x, args...) = sum(@broadcasted(logpdf(LogNormal(args...), x)))
@inline StudentT(nu, mu, sigma) = mu + sigma * TDist(nu)
@inline student_t_lpdf(x, args...) = sum(@broadcasted(logpdf(StudentT(args...), x)))
@inline cauchy_lpdf(x, location, scale) = sum(@broadcasted(logpdf(Cauchy(location, scale), x)))
@inline exponential_lpdf(x, args...) = sum(@broadcasted(logpdf(Exponential(args...), x)))
@inline double_exponential_lpdf(x, args...) = sum(@broadcasted(logpdf(DoubleExponential(args...), x)))
# https://mc-stan.org/docs/2_21/functions-reference/gamma-distribution.html
@inline gamma_lpdf(x, alpha, theta) = sum(@broadcasted(logpdf(Gamma(alpha, 1/theta), x)))
# https://mc-stan.org/docs/2_21/functions-reference/inverse-gamma-distribution.html
@inline inv_gamma_lpdf(x, alpha, theta) = sum(@broadcasted(logpdf(InverseGamma(alpha, theta), x)))
@inline uniform_lpdf(x, a, b) = sum(@broadcasted(logpdf(Uniform(a, b), x)))
@inline multi_normal_lpdf(x, mu, cov) = logpdf(MultivariateNormal(mu, cov), x)
@inline multi_normal_lpdf(x::AbstractMatrix, mu::AbstractVector, cov::AbstractMatrix) = sum(multi_normal_lpdf.(eachrow(x), Ref(mu), Ref(cov)))
@inline multi_normal_cholesky(x, mu, L) = error()
@inline log_sum_exp(args...) = logsumexp(args)
@inline log_sum_exp(x) = logsumexp(x)
@inline inv_logit(x) = logistic(x)
@inline log_inv_logit(x) = loglogistic(x)
@inline rep_vector(x, n) = fill(x, n)
@inline rep_matrix(x, args...) = fill(x, args)
@inline rep_row_vector(x, n) = fill(x, (1, n))
@inline append_col(x, y) = hcat(x, y)
@inline diag_matrix(x) = Diagonal(x)
@inline cholesky_decompose(x) = error()#cholesky(x).L
@inline square(x::Real) = x ^ 2
@inline pow(x, p) = x ^ p
@inline sd(x) = std(x)
@inline dot_self(x) = dot(x,x)
# https://mc-stan.org/docs/2_19/stan-users-guide/vectorizing-mixtures.html
@inline log_mix(lambda, lpdf1, lpdf2) = log_sum_exp(
    log(lambda) + lpdf1,
    log1m(lambda) + lpdf2
)
@inline constrain_(x, lower, upper) = if isfinite(lower) && isfinite(upper)
    sum(@broadcasted(log(upper - lower) - x - 2 * log1pexp(-x))), @.(lower + logistic(x) * (upper - lower))
elseif !isfinite(lower) && !isfinite(upper)
    0., x
elseif isfinite(lower) && !isfinite(upper)
    sum(x), @.(lower + exp(x))
else
    sum(x), @.(upper - exp(x))
end
# @inline constrain(x::AbstractVector; kwar)
@inline constrain(rv, x, lower=-Inf, upper=+Inf) = begin 
    drv, x = constrain_(x, lower, upper)
    rv + drv, x
end
# @inline matrix_constrain(rv, x)
@inline simplex_constrain(rv, x) = begin 
    drv, x = simplex_constrain(x)
    rv + drv, x
end
# logint(i::Integer) = log(Float64(i))::Float64
# https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
@inline simplex_constrain(y) = begin 
    # D = length(x)+1
    K = length(y)+1
    x = similar(y, K)
    drv = 0.
    rem = 1.
    for k in 1:length(y)
        z = logistic(y[k] - log(K - k))
        x[k] = rem * z
        drv += log(z) + log(1-z) + log(rem)
        rem -= x[k]
    end
    x[end] = rem
    drv, x
end
@inline ordered_constrain(rv, x) = begin 
    drv, x = ordered_constrain(x)
    rv + drv, x
end
@inline ordered_constrain(x) = sum(x[2:end]), cumsum(vcat(x[1], exp.(x[2:end])))
@inline positive_ordered_constrain(rv, x) = begin 
    drv, x = positive_ordered_constrain(x)
    rv + drv, x
end
@inline positive_ordered_constrain(x) = sum(x), cumsum(exp.(x))