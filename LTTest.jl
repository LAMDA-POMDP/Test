@everywhere using LaserTag

@everywhere function move_towards(b)
    s = typeof(b) <: LTState ? b : rand(b)
    # if typeof(b) <: LTInitialBelief
    #     s = rand(b)
    # elseif typeof(b) <: LTState
    #     s = b
    # else
    #     s = mode(b)
    # end
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
fl_bounds = AdaOPS.IndependentBounds(FORollout(move_towards_policy), 10, check_terminal=true, consistency_fix_thresh=1e-5)
pl_bounds = AdaOPS.IndependentBounds(PORollout(move_towards_policy, SIRParticleFilter(gen_lasertag(), 10)), 10, check_terminal=true, consistency_fix_thresh=1e-5)
flpu_bounds = AdaOPS.IndependentBounds(FORollout(move_towards_policy), POValue(QMDPSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)
flfu_bounds = AdaOPS.IndependentBounds(FORollout(move_towards_policy), FOValue(ValueIterationSolver(max_iterations=1000)), check_terminal=true, consistency_fix_thresh=1e-5)

# pomdp = gen_lasertag()
# b0 = initialstate(pomdp)
# solver = AdaOPSSolver(epsilon_0=0.1,
#                       bounds=flfu_bounds,
#                       rng=MersenneTwister(4),
#                       grid=nothing,
#                       k_min=3,
#                       zeta=0.3
#                      )
# p = solve(solver, pomdp)
# D, min_k = find_min_k(p, b0, 0.1)
# @show min_k
# tree_analysis(D)
# inchrome(D3Tree(D))

adaops_list = [:default_action=>[move_towards_policy,],
            :bounds=>[fl_bounds, pl_bounds, flpu_bounds, flfu_bounds],
            :delta=>[0.1, 0.3, 1.0],
            :grid=>[grid, nothing],
            :k_min=>[2, 3],
            :zeta=>[0.3, 0.4, 0.5]
            ]
adaops_list_labels = [["MoveTowards",],
                    ["(FO_MoveTowards, 10)", "(PO_MoveTowards, 10)", "(FO_MoveTowards, QMDP)", "(FO_MoveTowards, MDP)"],
                    [0.1, 0.3, 1.0],
                    ["FullGrid", "NullGrid"],
                    [2, 3],
                    [0.3, 0.4, 0.5]
                    ]
# For PL-DESPOT
bounds = PL_DESPOT.IndependentBounds(DefaultPolicyLB(move_towards_policy), 10, check_terminal=true, consistency_fix_thresh=1e-5)
pldespot_list = [:default_action=>[move_towards_policy,],
                    :bounds=>[bounds, ],
                    :K=>[100],
                    :lambda=>[0.01],
                    :C=>[Inf, 10., 20., 30.],
                    :beta=>[0.0, 0.1, 0.3]]
pldespot_list_labels = [["MoveTowards",],
                    ["(MoveTowards,10)", ],
                    [100],
                    [0.01],
                    [Inf, 10., 20., 30.],
                    [0.0, 0.1, 0.3]]

# For POMCPOW
value_estimator = FORollout(move_towards_policy)
pomcpow_list = [:estimate_value=>[value_estimator, ],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(100),],
                    :enable_action_pw=>[false,],
                    :k_observation=>[2.,],
                    :alpha_observation=>[0.15,]]
pomcpow_list_labels = [["MoveTowards", ],
                    [150000,],
                    [1.0,],
                    [100,],
                    [false,],
                    [2.,],
                    [0.15,]]


# Solver list
solver_list = [#PL_DESPOTSolver=>pldespot_list,
                AdaOPSSolver=>adaops_list,
                #POMCPOWSolver=>pomcpow_list,
                #QMDPSolver=>[:max_iterations=>[200,]],
                #FuncSolver=>[:func=>[move_towards,]],
                ]

solver_labels = [#"PL-DESPOT",
         "AdaOPS",
		 #"POMCPOW",
		 #"MoveTowards",
		 ]
solver_list_labels = [#pldespot_list_labels,
                adaops_list_labels,
                #pomcpow_list_labels,
                #[:max_iterations=>[200,]],
                #[["MoveTowards",]],
                ]

number_of_episodes = 100
max_steps = 100

parallel_experiment(gen_lasertag,
                    number_of_episodes,
                    max_steps,
                    solver_list,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    experiment_label="LT100",
                    full_factorial_design=true) 
