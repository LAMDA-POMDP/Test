using Pkg
Pkg.activate(".")

using LaserTag
# using Roomba
using POMDPs
using ParticleFilters
using POMDPSimulators
using QMDP
using POMDPPolicies
using Random
using Printf
using Distributions
using Plots

import Base.length
mutable struct StateGrid{D}
    convert::Function
    cutPoints::Vector{Vector{Float64}}

    function StateGrid{D}(convert, cutPoints...) where D
        newCutPoints = Array{Vector{Float64}}(undef, length(cutPoints))
        for i = 1:D
            if length(Set(cutPoints[i])) != length(cutPoints[i])
                error(@sprintf("Duplicates cutpoints are not allowed (duplicates observed in dimension %d)",i))
            end
            if !issorted(cutPoints[i])
                error("Cut points must be sorted")
            end
            newCutPoints[i] = cutPoints[i]
        end
        return new(convert, newCutPoints)
    end
end

StateGrid(convert, cutPoints...) = StateGrid{Base.length(cutPoints)}(convert, cutPoints...)
length(grid::StateGrid) = Base.length(grid.cutPoints)

zeros_like(grid::StateGrid) = zeros(Int, [length(points)+1 for points in grid.cutPoints]...)

function access(grid::StateGrid, access_cnt::Array, s, pomdp::POMDP)
    s = grid.convert(s, pomdp)::AbstractVector
    ind = zeros(Int, length(grid.cutPoints))
    for d in 1:length(grid.cutPoints)
        cutPoints = grid.cutPoints[d]
        # Binary search for the apt grid
        start_ind = 1
        end_ind = length(cutPoints) + 1
        mid_ind = floor(Int, (start_ind+end_ind)/2)
        while start_ind < end_ind
            cutPoint = cutPoints[mid_ind]
            if s[d] < cutPoint
                end_ind = mid_ind
            else
                start_ind = mid_ind + 1
            end
            mid_ind = floor(Int, (start_ind+end_ind)/2)
        end
        ind[d] = mid_ind
    end
    access_cnt[ind...] += 1
    return access_cnt[ind...] == 1 ? true : false
end

const MAX_SAMPLE_SIZE = 10000

function KLDSampleSize(k::Int, ζ::Float64 = 0.1, η::Float64 = 0.95)
"""
Return the minimum sample size in order to achieve an error at most ζ with a 95% level of confidence according to KLD-Sampling.
"""
    if k <= 1
        k = 1.2
    end
    a = (k-1)/2
    b = 1/(a*9)
    return (1-b+sqrt(b)*quantile(Normal(), η))^3*a/ζ
end

mutable struct WPFBelief{S} <: AbstractParticleBelief{S}
    particles::Vector{S}
    weights::Vector{Float64}
    weight_sum::Float64
    _probs::Union{Nothing, Dict{S,Float64}}
end

WPFBelief(particles::Vector{S}, weights::Vector{Float64}) where S = WPFBelief(particles, weights, sum(weights), nothing)
WPFBelief(particles::Vector{S}, weights::Vector{Float64}, weight_sum::Number) where S = WPFBelief(particles, weights, convert(Float64, weight_sum), nothing)

EffectiveSampleSize(b::WPFBelief) = b.weight_sum^2 / (b.weights'b.weights)

function ParticleFilters.rand(rng::AbstractRNG, b::WPFBelief)
    t = rand(rng) * b.weight_sum
    i = 1
    cum_weight = b.weights[1]
    while cum_weight < t
        i += 1
        cum_weight += b.weights[i]
    end
    return b.particles[i]
end

ParticleFilters.particles(b::WPFBelief) = b.particles
ParticleFilters.n_particles(b::WPFBelief) = length(b.particles)
ParticleFilters.weight(b::WPFBelief, i::Int) = b.weights[i]
ParticleFilters.particle(b::WPFBelief, i::Int) = b.particles[i]
ParticleFilters.weight_sum(b::WPFBelief) = b.weight_sum
ParticleFilters.weights(b::WPFBelief) = b.weights
ParticleFilters.weighted_particles(b::WPFBelief) = (b.particles[i]=>b.weights[i] for i in 1:length(b.weights))

function ParticleFilters.probdict(b::WPFBelief{S}) where S
    if b._probs === nothing
        # update the cache
        probs = Dict{S, Float64}()
        for (i,p) in enumerate(particles(b))
            if haskey(probs, p)
                probs[p] += weight(b, i)/weight_sum(b)
            else
                probs[p] = weight(b, i)/weight_sum(b)
            end
        end
        b._probs = probs
    end
    return b._probs
end

TVDistance(b1::AbstractParticleBelief, b2::AbstractParticleBelief, V) = sum(abs(pdf(b1,s)-pdf(b2,s)) for s in V) / 2.0

mutable struct AdaptiveParticleFilter{PM,RM,RS,RNG<:AbstractRNG} <: Updater
    predict_model::PM
    reweight_model::RM
    resampler::RS
    m_min::Int
    zeta::Float64
    MESS::Function
    ESS::Bool
    grid::Union{Nothing, StateGrid}
    rng::RNG
end

## Constructors ##
function AdaptiveParticleFilter(model, resampler, m_min::Integer; zeta::Float64=0.1, MESS::Function=KLDSampleSize, ESS::Bool=true, grid=nothing, rng::AbstractRNG=Random.GLOBAL_RNG)
    return AdaptiveParticleFilter(model, model, resampler, m_min, zeta, MESS, ESS, grid, rng)
end

function ParticleFilters.update(up::AdaptiveParticleFilter, b::WPFBelief, a, o)
    pm = particle_memory(up.predict_model)
    wm = Float64[]
    if EffectiveSampleSize(b) < n_particles(b) / 2.0
        RS = typeof(up.resampler)
        n = up.m_min
        curr_particle_num = 0
        if up.grid !== nothing
            access_cnt = zeros_like(up.grid)
        end
        k = 0
        likelihood_sum = 0.0
        likelihood_square_sum = 0.0
        ESS = 0.0
        while curr_particle_num < n
            resize!(pm, n)
            resize!(wm, n)
            resampled = ParticleFilters.resample(RS(n-curr_particle_num),
                            b,
                            up.predict_model,
                            up.reweight_model,
                            b, a, o,
                            up.rng)
            for (i, s) in enumerate(particles(resampled))
                sp, op = @gen(:sp, :o)(up.predict_model, s, a, up.rng)
                j = curr_particle_num + i
                pm[j] = sp
                wm[j] = pdf(observation(m, a, sp), o)
                likelihood_sum += wm[j]
                likelihood_square_sum += wm[j] * wm[j]
                if up.grid !== nothing
                    if access(up.grid, access_cnt, sp, up.predict_model)
                        k += 1
                    end
                end
            end
            if up.ESS
                if likelihood_sum == 0.0
                    curr_particle_num = n
                    n = min(curr_particle_num * 2, MAX_SAMPLE_SIZE)
                    continue
                end
                ESS = likelihood_sum .* likelihood_sum ./ likelihood_square_sum
            else
                ESS = n
            end
            if up.grid !== nothing
                MESS = up.MESS(k, up.zeta)
            else
                MESS = up.m_min
            end
            curr_particle_num = n
            n = min(ceil(Int, curr_particle_num*MESS/ESS), MAX_SAMPLE_SIZE)
        end
    else
        resize!(pm, n_particles(b))
        resize!(wm, n_particles(b))
        for (i, s) in enumerate(particles(b))
            if isterminal(up.predict_model, s)
                pm[i] = s
                wm[i] = 0.0
            else
                sp = @gen(:sp)(up.predict_model, s, a, up.rng)
                pm[i] = sp
                wm[i] = weight(b, i) * pdf(observation(m, a, sp), o)
            end
        end
    end
    return WPFBelief(pm, wm, sum(wm), nothing)
end

function POMDPs.initialize_belief(up::AdaptiveParticleFilter, b)
    pm = particle_memory(up.predict_model)
    wm = Float64[]
    n = up.m_min
    curr_particle_num = 0
    if up.grid !== nothing
        access_cnt = zeros_like(up.grid)
    end
    k = 0
    likelihood_sum = 0.0
    likelihood_square_sum = 0.0
    while curr_particle_num < n
        resize!(pm, n)
        resize!(wm, n)
        for i in (curr_particle_num+1):n
            pm[i] = rand(up.rng, b)
            wm[i] = 1.0
        end
        if up.grid !== nothing
            for s in pm[curr_particle_num+1:n]
                if access(up.grid, access_cnt, s, up.predict_model)
                    k += 1
                end
            end
        end
        likelihood_sum += sum(i for i in wm[curr_particle_num+1:n])
        likelihood_square_sum += sum(i*i for i in wm[curr_particle_num+1:n])
        if up.grid !== nothing
            MESS = up.MESS(k, up.zeta)
        else
            MESS = up.m_min
        end
        curr_particle_num = n
        n = min(ceil(Int, MESS), MAX_SAMPLE_SIZE)
    end
    WPFBelief(pm, wm)
end

function ParticleFilters.resample(r, bp::WPFBelief, predict_model, reweight_model, b, a, o, rng)
    for (i, s) in enumerate(particles(bp))
        if isterminal(predict_model, s)
            bp.weights[i] = 0.0
        end
    end
    bp.weight_sum = sum(bp.weights)
    if bp.weight_sum == 0.0
        error("Particle filter update error: all states in the particle collection were terminal.")
    end
    resample(r, bp, rng)
end

mutable struct SISFilter{PM,RM,RNG<:AbstractRNG,PMEM} <: Updater
    predict_model::PM
    reweight_model::RM
    n_init::Int
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

## Constructors ##
function SISFilter(model, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return SISFilter(model, model, n, rng)
end

function SISFilter(pmodel, rmodel, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return SISFilter(pmodel,
                    rmodel,
                    n,
                    rng,
                    particle_memory(pmodel),
                    Float64[]
                    )
end

function POMDPs.initialize_belief(up::SISFilter, b)
    pm = up._particle_memory
    wm = up._weight_memory
    resize!(pm, up.n_init)
    resize!(wm, up.n_init)
    for i in 1:up.n_init
        wm[i] = 1.0
        pm[i] = rand(b)
    end
    return WeightedParticleBelief(pm, wm)
end

function ParticleFilters.update(up::SISFilter, b::WeightedParticleBelief, a, o)
    pm = particle_memory(up.predict_model)
    wm = Float64[]
    resize!(pm, n_particles(b))
    resize!(wm, n_particles(b))
    for (i, s) in enumerate(particles(b))
        if isterminal(up.predict_model, s)
            pm[i] = s
            wm[i] = 0.0
        else
            sp = @gen(:sp)(up.predict_model, s, a, up.rng)
            pm[i] = sp
            wm[i] = weight(b, i) * pdf(observation(m, a, sp), o)
        end
    end
    return WeightedParticleBelief(pm, wm)
end

# function Random.seed!(f::AdaptiveParticleFilter, seed)
#     Random.seed!(f.rng, seed)
#     return f
# end

rng = MersenneTwister(10000)

# Laser Tag
n_row = 7
n_col = 11
n_obstacles = 8
m = gen_lasertag(n_row, n_col, n_obstacles, rng=rng)
convert(s::LTState, pomdp::LaserTagPOMDP) = [s.robot..., s.opponent...]
grid = StateGrid(convert, [2:3:n_row;], [2:3:n_col;], [2:3:n_row;], [2:3:n_col;])
resampler = (n)->LowVarianceResampler(n)

num_of_episodes = 1
rounds_per_episode = 500
max_step = 10
verbose = false

# policy = solve(RandomSolver(), m)
policy = solve(QMDPSolver(), m)
ground_truth_filter = BasicParticleFilter(m, resampler(1000000), 1000000)
V = states(m)

b0 = initialstate(m)
sir_particle_num = Float64[]
sir_tvd = Float64[]
sis_particle_num = Float64[]
sis_tvd = Float64[]
adare_particle_num = Float64[]
adare_tvd = Float64[]
ada_particle_num = Float64[]
ada_tvd = Float64[]
for i in 1:num_of_episodes
    println("$(i)-th episode")
    s = rand(b0)
    hist = []
    b_truth = initialize_belief(ground_truth_filter, b0)
    for i in 1:max_step
        if isterminal(m, s)
            break
        end
        a = action(policy, b_truth)
        o, s = @gen(:o, :sp)(m, s, a, rng)
        push!(hist, (a, o))
        b_truth = update(ground_truth_filter, b_truth, a, o)
    end

    println("SIR Particle Filter")
    for j in 1:rounds_per_episode
        if verbose
            println("$(j)-th round")
        end
        num_of_particles = ceil(Int, 10^(1+2*rand(rng)))
        sir_filter = BasicParticleFilter(m, resampler(num_of_particles), num_of_particles)
        b_sir = initialize_belief(sir_filter, b0)
        dist = 0.0
        try
            for (a, o) in hist
                b_sir = update(sir_filter, b_sir, a, o)
            end
            dist = TVDistance(b_truth, b_sir, V)
        catch
            continue
        end
        if isnan(dist)
            dist = 1.0
        end
        push!(sir_particle_num, num_of_particles)
        push!(sir_tvd, dist)
        if verbose
            println("Avg Particle Num: $(num_of_particles) Total Variation Distance: $(dist)")
        end
    end

    println("SIS Filter")
    for j in 1:rounds_per_episode
        if verbose
            println("$(j)-th round")
        end
        num_of_particles = ceil(Int, 10^(1+2*rand(rng)))
        sis_filter = SISFilter(m, num_of_particles)
        b_sis = initialize_belief(sis_filter, b0)
        dist = 0.0
        try
            for (a, o) in hist
                b_sis = update(sis_filter, b_sis, a, o)
            end
            dist = TVDistance(b_truth, b_sis, V)
        catch
            continue
        end
        if isnan(dist)
            dist = 1.0
        end
        push!(sis_particle_num, num_of_particles)
        push!(sis_tvd, dist)
        if verbose
            println("Avg Particle Num: $(num_of_particles) Total Variation Distance: $(dist)")
        end
    end

    println("Adaptive Resampling Particle Filters")
    for j in 1:rounds_per_episode
        if verbose
            println("$(j)-th round")
        end
        num_of_particles = ceil(Int, 10^(1+2*rand(rng)))
        adare_filter = AdaptiveParticleFilter(m, resampler(1000), num_of_particles, MESS=(x,y)->KLDSampleSize(x,y,0.99), ESS=false, grid=nothing, rng=rng)
        b_adare = initialize_belief(adare_filter, b0)
        step = 0
        dist = 0.0
        try
            for (a, o) in hist
                step += 1
                b_adare = update(adare_filter, b_adare, a, o)
            end
            dist = TVDistance(b_truth, b_adare, V)
        catch
            continue
        end
        if isnan(dist)
            dist = 1.0
        end
        push!(adare_particle_num, num_of_particles)
        push!(adare_tvd, dist)
        if verbose
            println("Avg Particle Num: $(num_of_particles) Total Variation Distance: $(dist)")
        end
    end

    println("Adaptive Particle Filters")
    for j in 1:rounds_per_episode
        if verbose
            println("$(j)-th round")
        end
        zeta = 10^(-2+1.6*rand())
        ada_filter = AdaptiveParticleFilter(m, resampler(1000), 10, zeta=zeta, MESS=(x,y)->KLDSampleSize(x,y,0.99), ESS=false, grid=grid, rng=rng)
        b_ada = initialize_belief(ada_filter, b0)
        step = 0
        dist = 0.0
        num_of_particles = 0
        try
            for (a, o) in hist
                step += 1
                b_ada = update(ada_filter, b_ada, a, o)
                num_of_particles += n_particles(b_ada)
            end
            dist = TVDistance(b_truth, b_ada, V)
        catch
            continue
        end
        if isnan(dist)
            dist = 1.0
        end
        push!(ada_particle_num, num_of_particles/step)
        push!(ada_tvd, dist)
        if verbose
            println("Avg Particle Num: $(num_of_particles/step) Total Variation Distance: $(dist)")
        end
    end
end
# plotly()
pyplot()

theme(:wong)
scatter(sis_particle_num, sis_tvd, label="SIS Particle Filter", markersize=4.0, markershape=:+)
scatter!(sir_particle_num, sir_tvd, label="SIR Particle Filter", markersize=4.0, markershape=:x)
# scatter!(adare_particle_num, adare_tvd, label="Adaptive Resampling Particle Filter", markersize=4.0, markershape=:circle)
scatter!(ada_particle_num, ada_tvd, label="Adaptive Particle Filter", markersize=4.0, markershape=:circle)

xlims!((10, 1000))
ylims!((0.0, 1.0))
xlabel!("Average Particle Number")
ylabel!("Total Variation Distance")
savefig("figures/particle_filter_comparison.svg")