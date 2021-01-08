using Pkg
Pkg.activate(".")

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
# using MCVI
using BasicPOMCP
using POMCPOW
using QMDP
using DiscreteValueIteration
using LocalApproximationValueIteration

using Random
using StaticArrays
using D3Trees
using CSV
using GridInterpolations
using LocalFunctionApproximation
using SubHunt

# Belief Updater
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
rng = Random.GLOBAL_RNG

# Low passive std may help agent identify useful information
info_gather_pomdp = SubHuntPOMDP(passive_std=0.5, ownspeed=5, passive_detect_radius=5, p_aware_kill=0.05)
pomdp = SubHuntPOMDP()

# Choose default domain
# m = info_gather_pomdp
# qmdp= solve(QMDPSolver(max_iterations=1000, verbose=true), m)
# mdp = solve(ValueIterationSolver(max_iterations=1000, verbose=true), UnderlyingMDP(m))
ping_first = PingFirst(qmdp)
qmdp_policy(p, b::AbstractParticleBelief) = value(qmdp, b)
POMDPs.action(p::PingFirst, b::SubHunt.SubHuntInitDist) = SubHunt.PING
POMDPs.action(p::AlphaVectorPolicy, s::SubState) = action(p, ParticleCollection([s]))

# For AdaOPS
convert(s::SubState, pomdp::SubHuntPOMDP) = SVector{3, Float64}(s.target..., s.goal)
grid = StateGrid(convert, range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=4, length=4)[2:end]
                            )

random_estimator = FORollout(RandomSolver())
# spl_bounds = AdaOPS.IndependentBounds(SemiPORollout(ping_first), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
bounds = AdaOPS.IndependentBounds(random_estimator, POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
# despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(ping_first), qmdp_policy, check_terminal=true, consistency_fix_thresh=1e-5)
despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), qmdp_policy, check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=100, tree_in_info=true, default_action=ping_first, bounds_warnings=false)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        grid=grid,
                        delta=2.0,
                        zeta=0.1,
                        m_init=30,
                        sigma=3,
                        bounds_warnings=false,
                        default_action=ping_first 
                        )
despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
# @time action(despot, b0)
# @time action(adaops, b0)
# show(stdout, MIME("text/plain"), info[:tree])
D, extra_info = build_tree_test(adaops, b0)
show(stdout, MIME("text/plain"), D)
extra_info_analysis(extra_info)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles), num_particles)
@show r = simulate(RolloutSimulator(), m, despot, belief_updater(m), b0, s0)
@show r = simulate(RolloutSimulator(), m, adaops, belief_updater(m), b0, s0)
let step = 1
    for (s, b, a, o) in stepthrough(m, despot, belief_updater(m), b0, s0, "s, b, a, o", max_steps=100)
        @show step
        @show s
        @show a
        step += 1
        # local D, extra_info = build_tree_test(p, b)
        # extra_info_analysis(extra_info)
    end
end
let step = 1
    for (s, b, a, o) in stepthrough(m, adaops, belief_updater(m), b0, s0, "s, b, a, o", max_steps=100)
        @show step
        @show s
        @show a
        step += 1
        # local D, extra_info = build_tree_test(p, b)
        # extra_info_analysis(extra_info)
    end
end
