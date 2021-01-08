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
using RockSample

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

function rsgen(map)
    possible_ps = [(i, j) for i in 1:map[1], j in 1:map[1]]
    selected = unique(rand(possible_ps, map[2]))
    while length(selected) != map[2]
        push!(selected, rand(possible_ps))
        selected = unique!(selected)
    end
    return RockSamplePOMDP(map_size=(map[1],map[1]), rocks_positions=selected)
end
struct MoveEast<:Policy end
POMDPs.action(p::MoveEast, b) = 2
move_east = MoveEast()


map = (11, 11)
# m = rsgen(map)
# qmdp_policy = solve(QMDPSolver(max_iterations=1000, verbose=true), m)
POMDPs.action(p::AlphaVectorPolicy, s::RSState) = action(p, ParticleCollection([s]))

b0 = initialstate(m)
s0 = rand(b0)

convert(s::RSState, pomdp::RockSamplePOMDP) = SVector(sum(s.rocks))
grid = StateGrid(convert, range(1, stop=map[2], length=map[2])[2:end])
po_bounds = AdaOPS.IndependentBounds(FORollout(move_east), POValue(qmdp_policy), check_terminal=true, consistency_fix_thresh=1e-5)

solver = AdaOPSSolver(bounds=po_bounds,
                        grid=grid,
                        delta=0.1,
                        zeta=0.1,
                        m_init=100,
                        sigma=6.0,
                        bounds_warnings=true,
                        default_action=move_east 
                        )

upper = (pomdp, b) -> value(qmdp_policy, b)
qmdp_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_east), upper, check_terminal=true, consistency_fix_thresh=1e-5)
despot_solver = DESPOTSolver(bounds=qmdp_bounds, K=100, tree_in_info=true, default_action=move_east, bounds_warnings=false)

despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
@time action(despot, b0)
@time action(adaops, b0)
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
