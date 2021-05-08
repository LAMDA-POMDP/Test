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

adaops_list = [
            :bounds=>[ada_blpu_bounds],
            :delta=>[0.1, 0.3, 1.0],
            :grid=>[grid],
            :m_min=>[10, 30, 100],
            :bounds_warnings=>[true],
            ]
adaops_list_labels = [
                    ["(Blind, QMDP)"],
                    [0.1, 0.3, 1.0],
                    ["FullGrid"],
                    [10, 30, 100],
                    [true],
                    ]

# For ARDESPOT
blpu_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(blind), ARDESPOT.FullyObservableValueUB(qmdp), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [
                    :default_action=>[move_towards_policy,], 
                    :bounds=>[blpu_bounds],
                    :K=>[30, 100, 300],
                    :lambda=>[0.0, 0.001, 0.01, 0.1],
                    :bounds_warnings=>[false],
                ]
ardespot_list_labels = [
                        ["MoveTowards",], 
                        ["(Blind, QMDP)"],
                        [30, 100, 300],
                        [0.0, 0.001, 0.01, 0.1],
                        [false],
                        ]

# For POMCPOW
value_estimator = FOValue(mdp)
pomcpow_list = [:estimate_value=>[value_estimator],
                    :tree_queries=>[150000,],
                    :max_time=>[1.0,],
                    :max_depth=>[90],
                    :criterion=>[MaxUCB(26.0),],
                    :final_criterion=>[MaxTries()],
                    :enable_action_pw=>[false],
                    :check_repeat_obs=>[false,],
                    :k_observation=>[4.,],
                    :alpha_observation=>[1/35,],
                    :tree_in_info=>[false],
                    :default_action=>[move_towards_policy],
                    ]
pomcpow_list_labels = [["MDP"],
                    [150000,],
                    [1.0,],
                    [90],
                    [26.0,],
                    ["MaxTries"],
                    [false],
                    [false],
                    [4.,],
                    [1/35,],
                    [false],
                    ["MoveTowards"],
                    ]

# Solver list
solver_list = [
                #DESPOTSolver=>ardespot_list,
                AdaOPSSolver=>adaops_list,
                #POMCPOWSolver=>pomcpow_list,
                # QMDPSolver=>[:max_iterations=>[1000,]],
                # FuncSolver=>[:func=>[move_towards,]],
                ]

solver_list_labels = [
                    #ardespot_list_labels,
                    adaops_list_labels,
                    #pomcpow_list_labels,
                    ]

solver_labels = [
                #"ARDESPOT",
                "AdaOPS",
                #"POMCPOW",
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
                    max_queue_length=18,
                    experiment_label="LT100_10",
                    full_factorial_design=true) 
