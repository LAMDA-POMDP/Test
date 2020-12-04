# Initialize multiple workers for parallel experiment
num_of_procs = 10 # You can also use addprocs() with no argument to create as many workers as your threads
using Distributed
addprocs(num_of_procs) # initial workers with the project env in current work directory

@everywhere push!(LOAD_PATH, "./ParallelExperiment/")
@everywhere using ParallelExp

# Make sure all your solvers are loaded in every procs
@everywhere using QMDP
@everywhere using POMCPOW

@everywhere push!(LOAD_PATH, "./LB-DESPOT/")
@everywhere using LBDESPOT # LB-DESPOT pkg
@everywhere push!(LOAD_PATH, "./UCT-DESPOT/")
@everywhere using UCTDESPOT # UCT-DESPOT pkg

# Make sure these pkgs are loaded in every procs
@everywhere using POMDPs # Basic POMDP framework
@everywhere using ParticleFilters # For simple particle filter
@everywhere using LaserTag

@everywhere using POMDPSimulators
@everywhere using POMDPPolicies
@everywhere using BeliefUpdaters

using Statistics
using DataFrames
using CSV
using Random
using Printf

pomdp = gen_lasertag()
belief_updater = SIRParticleFilter(pomdp, 20000)


@everywhere function move_towards(b)
    if typeof(b) <: LaserTag.LTInitialBelief
        return rand(1:5)
    elseif typeof(b) <: LBDESPOT.ScenarioBelief ||
        typeof(b) <: UCTDESPOT.ScenarioBelief ||
        typeof(b) <: ParticleFilters.ParticleCollection

        s = rand(b)
    else
        s = b
    end
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
move_towards_policy = solve(FunctionSolver(move_towards), pomdp)

# For LB-DESPOT
bounds = IndependentBounds(DefaultPolicyLB(move_towards_policy), 8.6, check_terminal=true)
random_bounds = IndependentBounds(DefaultPolicyLB(RandomPolicy(pomdp)), 200.0, check_terminal=true)
lbdespot_list = [:default_action=>[move_towards_policy,],
                    :bounds=>[bounds, random_bounds],
                    :K=>[500],
                    :lambda=>[0.01],
                    :beta=>[0.]]

# For UCT-DESPOT
rollout_policy = move_towards_policy
random_rollout_policy = RandomPolicy(pomdp)
uctdespot_list = [:default_action=>[move_towards_policy,],
                        :rollout_policy=>[rollout_policy, random_rollout_policy],
                        :K=>[3000, 5000],
                        :m=>[100, 1],
                        :c=>[100.,]]

# For POMCPOW
value_estimator = FORollout(move_towards_policy)
random_value_estimator = FORollout(RandomPolicy(pomdp))
pomcpow_list = [:estimate_value=>[value_estimator, random_value_estimator],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(100),],
                    :enable_action_pw=>[false,],
                    :k_observation=>[2.,],
                    :alpha_observation=>[0.15,]]

# Solver list
solver_list = [LB_DESPOTSolver=>lbdespot_list,
                #UCT_DESPOTSolver=>uctdespot_list,
                #POMCPOWSolver=>pomcpow_list,
                #QMDPSolver=>Dict(:max_iterations=>[200,]),
                #FuncSolver=>Dict(:func=>[move_towards,])]
                ]

number_of_episodes = 100
max_steps = 90

dfs = paralist[pomdp,
                          number_of_episodes,
                          max_steps, solver_list,
                          belief_updater=belief_updater,
                          full_factorial_design=false]

cd("./DESPOT_data")
CSV.write("LB_DESPOT.csv", dfs[1])
clist("[

println("finish")]