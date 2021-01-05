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
using VDPTag2

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
function VDPUpper(pomdp, b)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0
    else
        return mdp(cproblem(pomdp)).tag_reward
    end
end
rng = Random.GLOBAL_RNG

cpomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 1.8)))
pomdp = ADiscreteVDPTagPOMDP(cpomdp=cpomdp)
manage_uncertainty = ManageUncertainty(pomdp, 2.0)
to_next_ml = ToNextML(pomdp)
m = pomdp

# For AdaOPS
convert(s::TagState, pomdp) = s.target
grid = StateGrid(convert,
                range(-4, stop=4, length=5)[2:end-1],
                range(-4, stop=4, length=5)[2:end-1])
random_estimator = FORollout(RandomSolver())
flfu_bounds = AdaOPS.IndependentBounds(random_estimator, VDPUpper, check_terminal=true, consistency_fix_thresh=1e-5)
despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(manage_uncertainty), VDPUpper, check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=100, tree_in_info=true, default_action=manage_uncertainty, bounds_warnings=false)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=flfu_bounds,
                        grid=nothing,
                        delta=0.0,
                        zeta=0.4,
                        xi=0.95,
                        m_init=20,
                        sigma=3,
                        bounds_warnings=true,
                        default_action=manage_uncertainty 
                        )
despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
@time action(despot, b0)
@time action(adaops, b0)
# show(stdout, MIME("text/plain"), info[:tree])
D, extra_info = build_tree_test(adaops, b0)
# show(stdout, MIME("text/plain"), D)
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
