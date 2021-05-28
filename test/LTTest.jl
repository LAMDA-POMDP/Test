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
blind = BlindPolicySolver(max_iterations=1000)
qmdp = QMDPSolver(max_iterations=1000)
mdp = ValueIterationSolver(max_iterations=1000, include_Q=false)
@everywhere POMDPs.action(p::AlphaVectorPolicy, s::LTState) = action(p, ParticleCollection([s]))
@everywhere POMDPs.value(p::AlphaVectorPolicy, s::LTState) = action(p, ParticleCollection([s]))

# For AdaOPS
@everywhere Base.convert(::Type{SVector{2, Float64}}, s::LTState) = SVector{2,Float64}(s.opponent)
grid = StateGrid([2:7;], [2:11;])
ada_blpu_bounds = AdaOPS.IndependentBounds(POValue(blind), POValue(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

#
adaops_list = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.1],
            :grid=>[grid],
            :Deff_thres=>[2],
            :m_min=>[10],
            :bounds_warnings=>[true],
            ]
adaops_list_labels = [
                    ["(Blind, QMDP)"],
                    [0.1],
                    ["FullGrid"],
                    [2],
                    [10],
                    [true],
                    ]

#
adaops_list1 = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.0],
            :grid=>[grid],
            :Deff_thres=>[2],
            :m_min=>[30],
            :bounds_warnings=>[true],
            ]
adaops_list_labels1 = [
                    ["(Blind, QMDP)"],
                    [0.0],
                    ["FullGrid"],
                    [2],
                    [30],
                    [true],
                    ]
#
adaops_list2 = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.3],
            :grid=>[grid],
            :Deff_thres=>[0],
            :m_min=>[30],
            :bounds_warnings=>[true],
            ]
adaops_list_labels2 = [
                    ["(Blind, QMDP)"],
                    [0.3],
                    ["FullGrid"],
                    [0],
                    [30],
                    [true],
                    ]
#
adaops_list3 = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.1],
            :grid=>[StateGrid()],
            :Deff_thres=>[2],
            :m_min=>[100],
            :bounds_warnings=>[true],
            ]
adaops_list_labels3 = [
                    ["(Blind, QMDP)"],
                    [1.0],
                    ["Default"],
                    [2],
                    [100],
                    [true],
                    ]

# For ARDESPOT
blpu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(blind), ARDESPOT.FullyObservableValueUB(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [
                    :default_action=>[move_towards_policy,], 
                    :bounds=>[blpu_bounds],
                    :K=>[300],
                    :lambda=>[0.01],
                    :bounds_warnings=>[false],
                ]
ardespot_list_labels = [
                        ["MoveTowards",], 
                        ["(Blind, QMDP)"],
                        [300],
                        [0.01],
                        [false],
                        ]

# For POMCPOW
value_estimator = FOValue(mdp)
pomcpow_list = [:estimate_value=>[value_estimator],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :max_depth=>[90],
                    :criterion=>[MaxUCB(10.0),],
                    :final_criterion=>[MaxTries()],
                    :enable_action_pw=>[false],
                    :check_repeat_obs=>[false,],
                    :k_observation=>[4.,],
                    :alpha_observation=>[0.03,],
                    :tree_in_info=>[false],
                    :default_action=>[move_towards_policy],
                    ]
pomcpow_list_labels = [["MDP"],
                    [150000,],
                    [1.0,],
                    [90],
                    [10.0,],
                    ["MaxTries"],
                    [false],
                    [false],
                    [4.,],
                    [0.03,],
                    [false],
                    ["MoveTowards"],
                    ]

# Solver list
solver_list = [
                DESPOTSolver=>ardespot_list,
                AdaOPSSolver=>adaops_list,
                AdaOPSSolver=>adaops_list1,
                AdaOPSSolver=>adaops_list2,
                AdaOPSSolver=>adaops_list3,
                POMCPOWSolver=>pomcpow_list,
                ]

solver_list_labels = [
                    ardespot_list_labels,
                    adaops_list_labels,
                    adaops_list_labels1,
                    adaops_list_labels2,
                    adaops_list_labels3,
                    pomcpow_list_labels,
                    ]

solver_labels = [
                "ARDESPOT",
                "AdaOPS",
                "POMCPOW",
                ]

episodes_per_domain = 10
max_steps = 100 

parallel_experiment(gen_lasertag,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=100,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, LowVarianceResampler(30000), 30000),
                    max_queue_length=12,
                    experiment_label="LT100_10",
                    full_factorial_design=true) 
