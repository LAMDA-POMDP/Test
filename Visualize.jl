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

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
let k = 0
    global lidar_roomba_gen
    function lidar_roomba_gen()
        pomdp = RoombaPOMDP(sensor=Lidar(), mdp=RoombaMDP(config=k%3+1, aspace=action_space, v_max=max_speed))
        # k += 1
        return pomdp
    end
end
# Use lidar
pomdp = lidar_roomba_gen

# Belief updater
num_particles = 30000 # number of particles in belief
pos_noise_coeff = 0.3
ori_noise_coeff = 0.1
belief_updater = (m)->BasicParticleFilter(m, POMDPResampler(num_particles, LidarResampler(num_particles, m, pos_noise_coeff, ori_noise_coeff)), num_particles)

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

running = RunningSolver()
# For AdaOPS
convert(s::RoombaState, pomdp::RoombaPOMDP) = [s.x, s.y, s.theta]
grid = StateGrid(convert,
                range(-25, stop=15, length=10)[2:end-1],
                range(-20, stop=5, length=6)[2:end-1],
                range(0, stop=2*pi, length=4)[2:end-1])
flfu_bounds = AdaOPS.IndependentBounds(FORollout(running), FOValue(approx_solver), check_terminal=true)
splfu_bounds = AdaOPS.IndependentBounds(SemiPORollout(running), FOValue(approx_solver), check_terminal=true)
m = pomdp()
b0 = initialstate(m)
bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(running), ARDESPOT.FullyObservableValueUB(approx_solver), check_terminal=true)
# solver = DESPOTSolver(bounds=bounds, K=500, bounds_warnings=false, tree_in_info=true)
solver = AdaOPSSolver(bounds=splfu_bounds,
                        grid=grid,
                        delta=0.3,
                        zeta=0.1,
                        xi=0.99,
                        m_init=30,
                        enable_state_dict=true,
                        bounds_warnings=false
                        )
@time p = solve(solver, m)
@time action(p, b0)
# show(stdout, MIME("text/plain"), info[:tree])
D, extra_info = build_tree_test(p, b0)
show(stdout, MIME("text/plain"), D)
extra_info_analysis(extra_info)