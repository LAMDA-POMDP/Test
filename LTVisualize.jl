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
using LaserTag

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
function move_towards(pomdp, b)
    s = typeof(b) <: LTState ? b : rand(b)
    # try to sneak up diagonally
    diff = s.opponent-s.robot
    dx = diff[1]
    dy = diff[2]
    if abs(dx) == 0 && abs(dy) == 0
        LaserTag.DIR_TO_ACTION[[0, 0]]
    elseif abs(dx) < abs(dy)
        LaserTag.DIR_TO_ACTION[[0, sign(dy)]]
    else
        LaserTag.DIR_TO_ACTION[[sign(dx), 0]]
    end
end
move_towards_policy = FunctionPolicy(b->move_towards(nothing, b))

m = gen_lasertag()
qmdp = solve(QMDPSolver(max_iterations=1000, verbose=true), m)
POMDPs.action(p::AlphaVectorPolicy, s::LTState) = action(p, ParticleCollection([s]))

# For AdaOPS
convert(s::LTState, pomdp::LaserTagPOMDP) = s.opponent
grid = StateGrid(convert, [2:7;], [2:11;])
pu_bounds = AdaOPS.IndependentBounds(FORollout(RandomSolver()), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
bounds = AdaOPS.IndependentBounds(-20, POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
splpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), (p,b)->value(qmdp,b), check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=300, tree_in_info=true, default_action=move_towards_policy, bounds_warnings=false)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=bounds,
                        delta=0.3,
                        m_init=10,
                        sigma=8,
                        grid=grid,
                        zeta=0.005,
                        bounds_warnings=true,
                        default_action=move_towards_policy
                        )
despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
@time action(despot, b0)
@time action(adaops, b0)
# show(stdout, MIME("text/plain"), info[:tree])
D, extra_info = build_tree_test(adaops, b0)
show(stdout, MIME("text/plain"), D)
extra_info_analysis(D, extra_info)

num_particles = 30000
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles), num_particles)
@show simulate(RolloutSimulator(), m, despot, belief_updater(m), b0, s0)
@show simulate(RolloutSimulator(), m, adaops, belief_updater(m), b0, s0)
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
