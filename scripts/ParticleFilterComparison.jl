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
using AdaOPS
using Plots
using StaticArrays
using LinearAlgebra
using BeliefUpdaters

const MAX_SAMPLE_SIZE = 10000

EffectiveSampleSize(b::WeightedParticleBelief) = b.weight_sum^2 / dot(b.weights, b.weights)

TVDistance(b1::DiscreteBelief, b2::AbstractParticleBelief) = sum(abs(pdf(b1,s)-pdf(b2,s)) for s in b1.state_list) / 2.0

mutable struct AdaptiveParticleFilter{N,PM,RM,RS,RNG<:AbstractRNG} <: Updater
    predict_model::PM
    reweight_model::RM
    resampler::RS
    m_min::Int
    zeta::Float64
    Deff_thres::Float64
    min_occupied_bins::Int
    max_occupied_bins::Int
    grid::StateGrid{N}
    rng::RNG
end

## Constructors ##
function AdaptiveParticleFilter(model, resampler, m_min::Integer; min_occupied_bins::Int=2, zeta::Float64=KLDSampleSize(min_occupied_bins, m_min), grid::StateGrid=StateGrid(), max_occupied_bins::Int=prod(size(grid)), Deff_thres::Float64=2.0, rng::AbstractRNG=Random.GLOBAL_RNG)
    return AdaptiveParticleFilter(model, model, resampler, m_min, zeta, Deff_thres, min_occupied_bins, max_occupied_bins, grid, rng)
end

function ParticleFilters.update(up::AdaptiveParticleFilter{N}, b::WeightedParticleBelief, a, o) where N
    pm = particle_memory(up.predict_model)
    wm = Float64[]
    W = weights(b)
    w_sum = 0.0
    for (i, s) in enumerate(particles(b))
        if isterminal(up.predict_model, s)
            W[i] = 0.0
        else
            w_sum += W[i]
        end
    end
    b.weight_sum = w_sum
    if w_sum == 0.0
        error("All particles in the collection are impossible")
    end
    if n_particles(b) / EffectiveSampleSize(b) > up.Deff_thres
        RS = typeof(up.resampler)
        if N !== 0
            k = 0
            access_cnt = zeros_like(up.grid)
            for s in particles(b)
                k += access(grid, access_cnt, s, up.predict_model)
            end
            n = max(up.m_min, ceil(Int, KLDSampleSize(k, up.zeta)))
        end
        b = WeightedParticleBelief(particles(ParticleFilters.resample(RS(n),
                        b,
                        up.predict_model,
                        up.reweight_model,
                        b, a, o,
                        up.rng)), fill!(weights(b), 1.0), n)
    else
        n = n_particles(b)
    end
    resize!(pm, n)
    resize!(wm, n)
    W = weights(b)
    for (i, s) in enumerate(particles(b))
        if W[i] != 0.0
            pm[i] = @gen(:sp)(up.predict_model, s, a, up.rng)
            wm[i] = W[i] * pdf(observation(m, a, pm[i]), o)
        else
            pm[i] = s
            wm[i] = 0.0
        end
    end
    return WeightedParticleBelief(pm, wm, sum(wm))
end

function POMDPs.initialize_belief(up::AdaptiveParticleFilter{N}, b) where N
    pm = particle_memory(up.predict_model)
    wm = Float64[]
    if N !== 0
        n = ceil(Int, KLDSampleSize(up.max_occupied_bins, up.zeta))
    else
        n = up.m_min
    end
    resize!(pm, n)
    resize!(wm, n)
    for i in 1:n
        wm[i] = 1.0
        pm[i] = rand(b)
    end
    return WeightedParticleBelief(pm, wm)
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
    W = weights(b)
    w_sum = 0.0
    for (i, s) in enumerate(particles(b))
        if isterminal(up.predict_model, s)
            W[i] = 0.0
        else
            w_sum += W[i]
        end
    end
    if w_sum == 0.0
        error("All particles in the collection are impossible")
    end
    for (i, s) in enumerate(particles(b))
        if W[i] == 0.0
            pm[i] = s
            wm[i] = 0.0
        else
            sp = @gen(:sp)(up.predict_model, s, a, up.rng)
            pm[i] = sp
            wm[i] = W[i] * pdf(observation(m, a, sp), o)
        end
    end
    return WeightedParticleBelief(pm, wm)
end

# function Random.seed!(f::AdaptiveParticleFilter, seed)
#     Random.seed!(f.rng, seed)
#     return f
# end

rng = MersenneTwister(10000)
Random.seed!(rng, 10000)

# Laser Tag
n_row = 7
n_col = 11
n_obstacles = 8
m = gen_lasertag(n_row, n_col, n_obstacles, rng=rng)
Base.convert(::Type{SVector{4, Float64}}, s::LTState) = SVector{4,Float64}(s.robot..., s.opponent...)
# grid = StateGrid(2:3:n_row, 2:3:n_col, 2:3:n_row, 2:3:n_col)
grid = StateGrid(2:n_row, 2:n_col, 2:n_row, 2:n_col)
resampler = (n)->LowVarianceResampler(n)

num_of_episodes = 1
rounds_per_episode = 500
max_step = 10
verbose = false

get_floor(b::LaserTag.LTInitialBelief) = b.floor

function POMDPs.support(b::LaserTag.LTInitialBelief) # inefficient?
    if b.robot_init === nothing
        return LaserTag.states(b.floor)[1:end-1]
    else
        f = get_floor(b)
        vec = Array{LTState}(undef, LaserTag.n_pos(f))
        cnt = 0
        for i in 1:f.n_cols, j in 1:f.n_rows
            s = LTState(b.robot_init, Coord(i,j), false)
            cnt += 1
            vec[cnt] = s
        end
        return vec
    end
end

policy = solve(RandomSolver(), m)
# policy = solve(QMDPSolver(), m)
ground_truth_filter = DiscreteUpdater(m)
V = states(m)

b0 = initialstate(m)
sir_particle_num = Float64[]
sir_tvd = Float64[]
sis_particle_num = Float64[]
sis_tvd = Float64[]
ada_particle_num = Float64[]
ada_tvd = Float64[]
hist = []
for i in 1:num_of_episodes
    println("$(i)-th episode")
    s = rand(b0)
    global hist = []
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
            dist = TVDistance(b_truth, b_sir)
        catch ex
            @show ex
            continue
        end
        if isnan(dist)
            @warn("Total variation distance is not a number")
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
            dist = TVDistance(b_truth, b_sis)
        catch ex
            @show ex
            continue
        end
        if isnan(dist)
            @warn("Total variation distance is not a number")
            dist = 1.0
        end
        push!(sis_particle_num, num_of_particles)
        push!(sis_tvd, dist)
        if verbose
            println("Avg Particle Num: $(num_of_particles) Total Variation Distance: $(dist)")
        end
    end

    println("Adaptive Particle Filters")
    for j in 1:rounds_per_episode
        if verbose
            println("$(j)-th round")
        end
        dist = Beta(1, 1.6)
        zeta = 10^(-0.3+1.9*rand(rng, dist))
        ada_filter = AdaptiveParticleFilter(m, resampler(1000), 10, Deff_thres=1.6, zeta=zeta, grid=grid, rng=rng)
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
            dist = TVDistance(b_truth, b_ada)
        catch ex
            @show ex
            continue
        end
        if isnan(dist)
            @warn("Total variation distance is not a number")
            dist = 1.0
        end
        push!(ada_particle_num, num_of_particles/step)
        push!(ada_tvd, dist)
        if verbose
            @show zeta
            println("Avg Particle Num: $(num_of_particles/step) Total Variation Distance: $(dist)")
        end
    end
end
# plotly()
pyplot()

theme(:wong)
font_size = 12
scatter(sis_particle_num, sis_tvd, label="SIS Particle Filter", markersize=4.0, markershape=:+, xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
scatter!(sir_particle_num, sir_tvd, label="SIR Particle Filter", markersize=4.0, markershape=:x, xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
scatter!(ada_particle_num, ada_tvd, label="Adaptive Particle Filter", markersize=4.0, markershape=:circle, xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)

xlims!((10, 1000))
ylims!((0.0, 1.0))
xlabel!("Average Particle Number")
ylabel!("Total Variation Distance")
# savefig("figures/particle_filter_comparison.svg")