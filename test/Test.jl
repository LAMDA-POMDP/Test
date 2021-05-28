# global variables
max_workers = 6

# set up parallel environment
using Pkg
Pkg.activate(".")
using Distributed
addprocs(max_workers, exeflags="--project") # create n workers
cd("results")

@everywhere[
    using POMDPs
    using ParticleFilters
    using POMDPSimulators
    using POMDPPolicies
    using POMDPModelTools
    using BeliefUpdaters
    using ParallelExperiment

    # Solver
    using BSDESPOT
    using AdaOPS
    using ARDESPOT
    # using SARSOP
    # using PointBasedValueIteration
    using MCTS
    using BasicPOMCP
    using POMCPOW
    using QMDP
    using FIB
    using DiscreteValueIteration
    using LocalApproximationValueIteration
    using LocalApproximationRandomStrategy

    using SparseArrays
    using LinearAlgebra
    using StaticArrays

    using Random
]
using D3Trees
using CSV
using GridInterpolations
using LocalFunctionApproximation

# Belief Updater
@everywhere struct KLDResampler{D}
    grid::StateGrid{D}
    n_init::Int
    ζ::Float64
    η::Float64
end

@everywhere KLDResampler(grid, n_init=prod(size(grid)), ζ=0.01, η=0.01) = KLDResampler(grid, n_init, ζ, η)

@everywhere function ParticleFilters.resample(r::KLDResampler,
                                            bp::WeightedParticleBelief,
                                            pm::POMDP,
                                            rm::POMDP,
                                            b,
                                            a,
                                            o,
                                            rng)
    k = 0
    access_cnt = zeros_like(r.grid)
    P = particles(bp)
    W = weights(bp)
    w_sum = 0.0
    for (i, s) in enumerate(P)
        if isterminal(pm, s) || W[i] <= 0.0
            W[i] = 0.0
        else
            w_sum += W[i]
            k += access(r.grid, access_cnt, s, pm)
        end
    end
    if k == 0
        b0 = initialstate(pm)
        return ParticleCollection([rand(rng, b0) for i in 1:r.n_init])
    else
        bp = WeightedParticleBelief(P, W, w_sum)
        m = ceil(Int, KLDSampleSize(k, r.ζ, r.η))
        return resample(LowVarianceResampler(m), bp, rng)
    end
end

# init_param for AdaOPS

@everywhere function ParallelExperiment.init_param(m, bounds::AdaOPS.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper = init_param(m, bounds.upper)
    AdaOPS.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function ParallelExperiment.init_param(m, bound::PORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    if typeof(bound.updater) <: BasicParticleFilter
        PORollout(policy, BasicParticleFilter(m,
                                                bound.updater.resampler,
                                                bound.updater.n_init,
                                                bound.updater.rng))
    else
        PORollout(policy, bound.updater)
    end
end

@everywhere function ParallelExperiment.init_param(m, bound::FORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    FORollout(policy)
end

@everywhere function ParallelExperiment.init_param(m, bound::SemiPORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    SemiPORollout(policy)
end

@everywhere function ParallelExperiment.init_param(m, bound::POValue)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    POValue(policy)
end

@everywhere function ParallelExperiment.init_param(m, bound::FOValue)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, UnderlyingMDP(m)) : bound.solver
    FOValue(policy)
end

# init_param for BSDESPOT

@everywhere function ParallelExperiment.init_param(m, bounds::BSDESPOT.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper_policy = init_param(m, bounds.upper)
    if typeof(upper_policy) <: Policy
        upper = (p, b)->value(upper_policy, b)
    else
        upper = upper_policy
    end
    BSDESPOT.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function ParallelExperiment.init_param(m, bound::BSDESPOT.FullyObservableValueUB)
    policy = typeof(bound.p) <: Solver ? solve(bound.p, UnderlyingMDP(m)) : bound.p
    BSDESPOT.FullyObservableValueUB(policy)
end

# init_param for ARDESPOT

@everywhere function ParallelExperiment.init_param(m, bounds::ARDESPOT.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper_policy = init_param(m, bounds.upper)
    if typeof(upper_policy) <: Policy
        upper = (p, b)->value(upper_policy, b)
    else
        upper = upper_policy
    end
    ARDESPOT.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function ParallelExperiment.init_param(m, bound::ARDESPOT.FullyObservableValueUB)
    if typeof(bound.p) <: QMDPSolver || typeof(bound.p) <: RSQMDPSolver
        policy = solve(bound.p, m)
    elseif typeof(bound.p) <: Solver
        policy = solve(bound.p, UnderlyingMDP(m))
    else
        policy = bound.p
    end
    ARDESPOT.FullyObservableValueUB(policy)
end

@everywhere function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

@everywhere function lower_bounded_zeta(d, k, zeta=0.8)
    max(zeta, 1 - (0.2*k + 0.2*(1-d)))
end

# include("LidarRoombaTest.jl")
# include("BumperRoombaTest.jl")
# include("RSTest.jl")
# include("LTTest.jl")
include("LightDarkTest.jl")

# Not ready yet:
# include("VDPTagTest.jl")
# include("SHTest.jl")