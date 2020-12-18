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
using Roomba

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

max_speed = 10.0
speed_gears = 3
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om*max_turn_rate/v) for v in range(1, stop=max_speed, length=speed_gears) for om in [-1, 1]])
m = RoombaPOMDP(sensor=Bumper(), mdp=RoombaMDP(config=1, aspace=action_space, v_max=max_speed))

# Belief updater
num_particles = 30000 # number of particles in belief
pos_noise_coeff = 0.3
ori_noise_coeff = 0.1
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles, BumperResampler(num_particles, m, pos_noise_coeff, ori_noise_coeff)), num_particles)

#grid = RectangleGrid(range(-25, stop=15, length=201),
#                    range(-20, stop=5, length=101),
#                    range(0, stop=2*pi, length=61),
#                    range(0, stop=1, length=2)) # Create the interpolating grid
grid = RectangleGrid(range(-25, stop=15, length=21),
                    range(-20, stop=5, length=11),
                    range(0, stop=2*pi, length=6),
                    range(0, stop=1, length=2)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000)

mdp = solve(approx_solver, m)

# For AdaOPS
running = solve(RunningSolver(), m)
convert(s::RoombaState, pomdp::RoombaPOMDP) = [s.x, s.y, s.theta]
grid = StateGrid(convert,
                range(-25, stop=15, length=7)[2:end-1],
                range(-20, stop=5, length=5)[2:end-1],
                range(0, stop=2*pi, length=4)[2:end-1])
flfu_bounds = AdaOPS.IndependentBounds(FORollout(running), FOValue(mdp), check_terminal=true)
splfu_bounds = AdaOPS.IndependentBounds(SemiPORollout(running), FOValue(mdp), check_terminal=true)
bounds = AdaOPS.IndependentBounds(SemiPORollout(running), 10.0, check_terminal=true)
despot_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(running), ARDESPOT.FullyObservableValueUB(approx_solver), check_terminal=true)
despot_solver = DESPOTSolver(bounds=despot_bounds, K=100, bounds_warnings=false, tree_in_info=true, default_action=running)
b0 = initialstate(m)
s0 = rand(b0)
solver = AdaOPSSolver(bounds=splfu_bounds, #                         grid=grid,
                        delta=0.1,
                        zeta=0.3,
                        xi=0.95,
                        m_init=30,
                        m_min=0.2,
                        m_max=20.0,
                        bounds_warnings=false
                        )
despot = solve(despot_solver, m)
adaops = solve(solver, m)
# @time p = solve(solver, m)
# @time action(p, b0)
# show(stdout, MIME("text/plain"), info[:tree])
# D, extra_info = build_tree_test(p, b0)
# show(stdout, MIME("text/plain"), D)
# extra_info_analysis(extra_info)

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
        if o == true
            println("Wall Contact!")
        end
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
        if o == true
            println("Wall Contact!")
        end
    end
end
