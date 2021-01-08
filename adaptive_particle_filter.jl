using AdaOPS
using LaserTag
using POMDPs
using ParticleFilters
using POMDPSimulators
using Random

mutable struct WPFBelief{S} <: AbstractParticleBelief{S}
    particles::Vector{S}
    weights::Vector{Float64}
    weight_sum::Float64
    _probs::Union{Nothing, Dict{S,Float64}}
end

WPFBelief(particles::Array, weights::Array{Float64,1}) = WPFBelief(particles, weights, sum(weights), nothing)
WPFBelief(particles::Array, weights::Array{Float64,1}, weight_sum::Number) = WPFBelief(particles, weights, convert(Float64, weight_sum), nothing)

ParticleFilters.rand(rng::R, b::WPFBelief) where R<:AbstractRNG = rand(rng, b.particles)
ParticleFilters.particles(b::WPFBelief) = b.particles
ParticleFilters.n_particles(b::WPFBelief) = length(b.particles)
ParticleFilters.weight(b::WPFBelief, i::Int) = b.weights[i]
ParticleFilters.particle(b::WPFBelief, i::Int) = b.particles[i]
ParticleFilters.weight_sum(b::WPFBelief) = b.weight_sum
ParticleFilters.weights(b::WPFBelief) = b.weights
ParticleFilters.weighted_particles(b::WPFBelief) = (b.particles[i]=>b.weights[i] for i in 1:length(b.weights))

function ParticleFilters.probdict(b::WPFBelief{S, O}) where S where O
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

mutable struct AdaptiveParticleFilter{PM,RM,RS,RNG<:AbstractRNG,PMEM} <: Updater
    predict_model::PM
    reweight_model::RM
    resampler::RS
    n_init::Int
    zeta::Float64
    MESS::Function
    ESS::Bool
    grid::Union{Nothing, StateGrid}
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

## Constructors ##
function AdaptiveParticleFilter(model, resampler, n_init::Integer, zeta::Float64=0.1, MESS::Function=KLDSampleSize, grid=nothing, ESS=true, rng::AbstractRNG=Random.GLOBAL_RNG)
    return AdaptiveParticleFilter(model, model, resampler, n_init, zeta, MESS, grid, ESS, rng)
end

function AdaptiveParticleFilter(pmodel, rmodel, resampler, n_init::Integer, zeta::Float64=0.1, MESS::Function=KLDSampleSize, grid=nothing, ESS=true, rng::AbstractRNG=Random.GLOBAL_RNG)
    return AdaptiveParticleFilter(pmodel,
                               rmodel,
                               resampler,
                               n_init,
                               zeta,
                               MESS,
                               ESS,
                               grid,
                               rng,
                               particle_memory(pmodel),
                               Float64[]
                              )
end

function ParticleFilters.update(up::AdaptiveParticleFilter, b::WPFBelief, a, o)
    RS = typeof(up.resampler)
    n = up.n_init
    curr_particle_num = 0
    if up.grid !== nothing
        access_cnt = zeros_like(up.grid)
    end
    k = 0
    likelihood_sum = 0.0
    likelihood_square_sum = 0.0
    pm = up._particle_memory
    wm = up._weight_memory
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
        ParticleFilters.predict!(view(pm, curr_particle_num+1:n), up.predict_model, resampled, a, o, up.rng)
        ParticleFilters.reweight!(view(wm, curr_particle_num+1:n), up.reweight_model, resampled, a, pm, o, up.rng)
        if p.sol.grid !== nothing
            for sp in pm[curr_particle_num+1:n]
                if access(up.grid, access_cnt, sp, up.predict_model)
                    k += 1
                end
            end
        end
        likelihood_sum += sum(i in wm[curr_particle_num+1:n])
        likelihood_square_sum += sum(i*i in wm[curr_particle_num+1:n])
        if up.ESS
            ESS += likelihood_sum .* likelihood_sum ./ likelihood_square_sum
        else
            ESS += n - curr_particle_num
        end
        if p.sol.grid !== nothing
            MESS = up.MESS(k, up.zeta)
        else
            MESS = up.n_init
        end
        curr_particle_num = n
        n = ceil(Int, curr_particle_num*MESS/ESS)
    end
    return WPFBelief(pm, wm, sum(wm), nothing)
end

function Random.seed!(f::AdaptiveParticleFilter, seed)
    Random.seed!(f.rng, seed)
    return f
end

function ParticleFilters.resample(resampler, bp::WPFBelief, pm::Union{POMDP,MDP}, rm, b, a, o, rng)
    weightSum = 0.0
    for (i,s) in enumerate(particles(bp))
        if isterminal(pm, s)
            weights(bp)[i] = 0.0
        else
            weightSum += weight(bp, i)
        end
    end
    if weightSum == 0.0
        error("Particle filter update error: all states in the particle collection are terminal.")
    end
    resample(resampler, bp, rng)
end

struct POMDPResampler{R}
    n::Int
    r::R
end

POMDPResampler(n, r=LowVarianceResampler(n)) = POMDPResampler(n, r)

function ParticleFilters.resample(r::POMDPResampler,
                                  bp::WeightedParticleBelief,
                                  pm::POMDP,
                                  rm::POMDP,
                                  b,
                                  a,
                                  o,
                                  rng)

    if weight_sum(bp) == 0.0
        # no appropriate particles - resample from the initial distribution
        new_ps = [rand(rng, initialstate(pm)) for i in 1:r.n]
        return ParticleCollection(new_ps)
    else
        # normal resample
        return resample(r.r, bp, rng)
    end
end

n_row = 20
n_col = 20
m = gen_lasertag(n_rows=n_row,
                n_cols=n_col,
                obstacles=gen_obstacles(n_row, n_col, 32, rng))
convert(s::LTState, pomdp::LaserTagPOMDP) = [s.robot..., s.opponent...]
grid = StateGrid(convert, [2:20;], [2:20;], [2:20;], [2:20;])
nums = [100, 300, 1000, 3000, 10000, 30000]
for i in 1:6
    num_of_particles = nums[i]
    ground_truth = BasicParticleFilter(m, resampler, 100000)
    resampler = POMDPResampler(num_of_particles, LowVarianceResampler(num_of_particles))
    sir_filter = BasicParticleFilter(m, resampler, num_of_particles)
    kld_filter = AdaptiveParticleFilter(pomdp, resampler, num_of_particles/10, zeta=0.01, MESS=(x,y)->KLDSampleSize(x,y,0.99), grid=grid, ESS=false)
    ada_filter = AdaptiveParticleFilter(pomdp, resampler, num_of_particles/10, zeta=0.01, MESS=(x,y)->KLDSampleSize(x,y,0.99), grid=grid, ESS=true)
end