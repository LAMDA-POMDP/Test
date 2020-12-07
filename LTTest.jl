@everywhere using LaserTag

@everywhere function move_towards(b)
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
move_towards_policy = FunctionPolicy(b->move_towards(b))

# For AdaOPS
@everywhere POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::LTState, pomdp::LaserTagPOMDP) = s.opponent
grid = StateGrid([2:7;], [2:11;])
flpu_bounds = AdaOPS.IndependentBounds(FORollout(move_towards_policy), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
pl10pu_bounds = AdaOPS.IndependentBounds(PORollout(move_towards_policy, SIRParticleFilter(gen_lasertag(), 10)), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
pl30pu_bounds = AdaOPS.IndependentBounds(PORollout(move_towards_policy, SIRParticleFilter(gen_lasertag(), 30)), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
flfu_bounds = AdaOPS.IndependentBounds(FORollout(move_towards_policy), FOValue(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

# pomdp = gen_lasertag()
# b0 = initialstate(pomdp)
# solver = AdaOPSSolver(epsilon_0=0.1,
#                       bounds=flpu_bounds,
#                       rng=MersenneTwister(4),
#                       grid=grid,
#                       k_min=2,
#                       zeta=0.5
#                      )
# @time p = solve(solver, pomdp)
# D, extra_info = build_tree_test(p, b0)
# extra_info_analysis(extra_info)
# inchrome(D3Tree(D))

adaops_list = [:default_action=>[move_towards_policy,],
            :bounds=>[flpu_bounds, flfu_bounds, pl10pu_bounds, pl30pu_bounds],
            :delta=>[0.1, 0.3, 1.0],
            :grid=>[grid, nothing],
            :k_min=>[2],
            :zeta=>[0.1, 0.3, 0.5],
            :xi=>[0.95, 0.3, 0.1],
            :bounds_warnings=>[false],
            ]
adaops_list_labels = [["MoveTowards",],
                    ["(FO_MoveTowards, QMDP)", "(FO_MoveTowards, MDP)", "(PO_MoveTowards_10, MDP)", "(PO_MoveTowards_30, MDP)"],
                    [0.1, 0.3, 1.0],
                    ["FullGrid", "NullGrid"],
                    [2],
                    [0.1, 0.3, 0.5],
                    [0.95, 0.3, 0.1],
                    [false],
                    ]
# For PL-DESPOT
flfu_bounds = PL_DESPOT.IndependentBounds(PL_DESPOT.DefaultPolicyLB(move_towards_policy), PL_DESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)
pldespot_list = [:default_action=>[move_towards_policy,],
                    :bounds=>[flfu_bounds],
                    :K=>[100],
                    :lambda=>[0.01],
                    :C=>[Inf, 10., 20., 30.],
                    :beta=>[0.0, 0.1, 0.3],
                    :bounds_warnings=>[false],
                    ]
pldespot_list_labels = [["MoveTowards",],
                    ["(MoveTowards, MDP)"],
                    [100],
                    [0.01],
                    [Inf, 10., 20., 30.],
                    [0.0, 0.1, 0.3],
                    [false],
                    ]

# For ARDESPOT
bounds_ub = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_towards_policy), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [:default_action=>[move_towards_policy,], 
                    :bounds=>[bounds_ub],
                    :K=>[100, 300],
                    :lambda=>[0.0, 0.01, 0.1],
                    :bounds_warnings=>[false],
                ]
ardespot_list_labels = [["MoveTowards",], 
                        ["(MoveTowards, 10)", "(MoveTowards, MDP)"],
                        [100, 300],
                        [0.0, 0.01, 0.1],
                        [false],
                        ]

# For POMCPOW
value_estimator = FORollout(move_towards_policy)
random_value_estimator = FORollout(RandomSolver())
pomcpow_list = [:estimate_value=>[value_estimator, random_value_estimator],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(100),],
                    :enable_action_pw=>[false, true],
                    :k_observation=>[2.,],
                    :alpha_observation=>[0.15,]]
pomcpow_list_labels = [["MoveTowardsRollout", "RandomRollout"],
                    [150000,],
                    [1.0,],
                    [100,],
                    [false, true],
                    [2.,],
                    [0.15,]]


# Solver list
solver_list = [
                PL_DESPOTSolver=>pldespot_list,
                DESPOTSolver=>ardespot_list,
                AdaOPSSolver=>adaops_list,
                POMCPOWSolver=>pomcpow_list,
                QMDPSolver=>[:max_iterations=>[1000,]],
                FuncSolver=>[:func=>[move_towards,]],
                ]

solver_labels = [
                "PL-DESPOT",
                "ARDESPOT",
                "AdaOPS",
                "POMCPOW",
                "QMDP",
                "MoveTowards",
                ]
solver_list_labels = [
                    pldespot_list_labels,
                    ardespot_list_labels,
                    adaops_list_labels,
                    pomcpow_list_labels,
                    [[1000,]],
                    [["MoveTowards",]],
                    ]

number_of_episodes = 1000
max_steps = 100

parallel_experiment(gen_lasertag,
                    number_of_episodes,
                    max_steps,
                    solver_list,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=300,
                    experiment_label="LT1000",
                    full_factorial_design=true) 
