@everywhere using LaserTag

@everywhere function move_towards(pomdp, b)
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
qmdp = QMDPSolver(max_iterations=1000)
mdp = ValueIterationSolver(max_iterations=1000, include_Q=false)

# For AdaOPS
convert(s::LTState, pomdp::LaserTagPOMDP) = s.opponent
grid = StateGrid(convert, [2:7;], [2:11;])
pu_bounds = AdaOPS.IndependentBounds(-20.0, POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
splmpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(move_towards_policy), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)
splqpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

# pomdp = gen_lasertag()
# b0 = initialstate(pomdp)
# solver = AdaOPSSolver(bounds=pu_bounds,
#                       grid=grid,
#                       m_init=30,
#                       zeta=0.4
#                      )
# @time p = solve(solver, pomdp)
# D, extra_info = build_tree_test(p, b0)
# extra_info_analysis(extra_info)
# inchrome(D3Tree(D))

adaops_list = [:default_action=>[move_towards_policy,],
            :bounds=>[pu_bounds, splmpu_bounds, splqpu_bounds],
            :delta=>[0.1, 0.3, 1.0],
            :grid=>[grid],
            :m_init=>[30],
            :zeta=>[0.1, 0.3],
            :xi=>[0.95, 0.3],
            :bounds_warnings=>[false],
            ]
adaops_list_labels = [["MoveTowards",],
                    ["(-20, QMDP)", "(SemiPO_MoveTowards, QMDP)", "(SemiPO_QMDP, QMDP)"],
                    [0.1, 0.3, 1.0],
                    ["FullGrid"],
                    [30],
                    [0.1, 0.3],
                    [0.95, 0.3],
                    [false],
                    ]

# For ARDESPOT
flfu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(move_towards_policy), ARDESPOT.FullyObservableValueUB(mdp), check_terminal=true, consistency_fix_thresh=1e-5)
plfu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(qmdp), ARDESPOT.FullyObservableValueUB(mdp), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [:default_action=>[move_towards_policy,], 
                    :bounds=>[flfu_bounds, plfu_bounds],
                    :K=>[100, 300],
                    :bounds_warnings=>[false],
                ]
ardespot_list_labels = [["MoveTowards",], 
                        ["(MoveTowards, MDP)", "(QMDPRollout, MDP)"],
                        [100, 300],
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
                DESPOTSolver=>ardespot_list,
                AdaOPSSolver=>adaops_list,
                POMCPOWSolver=>pomcpow_list,
                # QMDPSolver=>[:max_iterations=>[1000,]],
                # FuncSolver=>[:func=>[move_towards,]],
                ]

solver_labels = [
                "ARDESPOT",
                "AdaOPS",
                "POMCPOW",
                # "QMDP",
                # "MoveTowards",
                ]
solver_list_labels = [
                    ardespot_list_labels,
                    adaops_list_labels,
                    pomcpow_list_labels,
                    # [[1000,]],
                    # [["MoveTowards",]],
                    ]

episodes_per_domain = 100
max_steps = 100

parallel_experiment(gen_lasertag,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=10,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=300,
                    experiment_label="LT10*100",
                    full_factorial_design=true) 
