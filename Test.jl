# global variables
max_workers = 64

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
    using PL_DESPOT
    using AdaOPS
    using ARDESPOT
    # using SARSOP
    # using PointBasedValueIteration
    using MCTS
    using BasicPOMCP
    using POMCPOW
    using QMDP
    using DiscreteValueIteration
    using LocalApproximationValueIteration

    using LinearAlgebra
    using StaticArrays

    using Random
]
using D3Trees
using CSV
using GridInterpolations
using LocalFunctionApproximation

# Belief Updater
@everywhere struct POMDPResampler{R}
    n::Int
    r::R
end

@everywhere POMDPResampler(n, r=LowVarianceResampler(n)) = POMDPResampler(n, r)

@everywhere function ParticleFilters.resample(r::POMDPResampler,
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

# init_param for different solvers

@everywhere function ParallelExperiment.init_param(m, bounds::AdaOPS.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper = init_param(m, bounds.upper)
    AdaOPS.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh, bounds.max_depth)
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

@everywhere function ParallelExperiment.init_param(m, bounds::PL_DESPOT.IndependentBounds)
    lower = init_param(m, bounds.lower)
    if typeof(bounds.upper) <: QMDPSolver
        qmdp = solve(bounds.upper, m)
        upper = (p, b)->value(qmdp, b)
    else
        upper = init_param(m, bounds.upper)
    end
    PL_DESPOT.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function ParallelExperiment.init_param(m, bound::PL_DESPOT.FullyObservableValueUB)
    policy = typeof(bound.p) <: Solver ? solve(bound.p, UnderlyingMDP(m)) : bound.p
    PL_DESPOT.FullyObservableValueUB(policy)
end

@everywhere function ParallelExperiment.init_param(m, bounds::ARDESPOT.IndependentBounds)
    lower = init_param(m, bounds.lower)
    if typeof(bounds.upper) <: QMDPSolver
        qmdp = solve(bounds.upper, m)
        upper = (p, b)->value(qmdp, b)
    else
        upper = init_param(m, bounds.upper)
    end
    ARDESPOT.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function ParallelExperiment.init_param(m, bound::ARDESPOT.FullyObservableValueUB)
    policy = typeof(bound.p) <: Solver ? solve(bound.p, UnderlyingMDP(m)) : bound.p
    ARDESPOT.FullyObservableValueUB(policy)
end

@everywhere function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::WPFBelief)
    util = zeros(length(p.alphas))
    for (i, s) in enumerate(particles(b))
        j = stateindex(p.pomdp, s)
        util += weight(b, i)*getindex.(p.alphas, (j,))
    end
    return util
end

@everywhere struct ModePolicy <: Policy
    policy::Policy
end
@everywhere struct ModeSolver{P<:Union{Solver,Policy}} <: Solver
    solver::P
end
@everywhere POMDPs.solve(solver::ModeSolver{P}, pomdp) where P <: Solver = ModePolicy(solve(solver.solver, pomdp))
@everywhere POMDPs.solve(solver::ModeSolver{P}, pomdp) where P <: Policy = ModePolicy(solver.solver)
@everywhere POMDPs.action(p::ModePolicy, b::AbstractParticleBelief) = action(p.policy, mode(b))
@everywhere POMDPs.action(p::ModePolicy, b) = action(p.policy, b)

@everywhere struct RandPolicy <: Policy
    policy::Policy
end
@everywhere struct RandSolver{P<:Union{Solver,Policy}} <: Solver
    solver::P
end
@everywhere POMDPs.solve(solver::RandSolver{P}, pomdp) where P <: Solver = RandPolicy(solve(solver.solver, pomdp))
@everywhere POMDPs.solve(solver::RandSolver{P}, pomdp) where P <: Policy = RandPolicy(solver.solver)
@everywhere POMDPs.action(p::RandPolicy, b::AbstractParticleBelief) = action(p.policy, rand(b))
@everywhere POMDPs.action(p::RandPolicy, b) = action(p.policy, b)

# include("LidarRoombaTest.jl")
include("BumperRoombaTest.jl")
# include("VDPTagTest.jl")
# include("SHTest.jl")
# include("RSTest.jl")
# include("LTTest.jl")
# include("BabyTest.jl")