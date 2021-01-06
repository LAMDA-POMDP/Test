@everywhere using SubHunt

# Low passive std may help agent identify useful information
info_gather_pomdp = SubHuntPOMDP(passive_std=0.5, ownspeed=5, passive_detect_radius=5, p_aware_kill=0.05)
pomdp = SubHuntPOMDP()

# Choose default domain
m = pomdp
qmdp= solve(QMDPSolver(max_iterations=1000), m)
mdp = solve(ValueIterationSolver(max_iterations=1000), m)
ping_first = PingFirst(qmdp)

@everywhere convert(s::SubState, pomdp::SubHuntPOMDP) = SVector{3, Float64}(s.target..., s.goal)
grid = StateGrid(convert, range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=4, length=4)[2:end]
                            )
POMDPs.action(p::PingFirst, b::SubHunt.SubHuntInitDist) = SubHunt.PING

flpu_bounds = AdaOPS.IndependentBounds(FORollout(mdp), POValue(qmdp_policy))
plpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(ping_first), POValue(qmdp_policy))

adaops_list = [:default_action=>[ping_first,],
            :bounds=>[flfu_bounds, plpu_bounds],
            :delta=>[0.1, 0.3],
            :grid=>[nothing, grid],
            :m_init=>[30, 50],
            :zeta=>[0.1, 0.3],
            ]
adaops_list_labels = [["PingFirst",],
                    ["(FO_PingFirst, QMDP)", "(PO_PingFirst, QMDP)",],
                    [0.1, 0.3],
                    ["NullGrid", "FullGrid"],
                    [30, 50],
                    [0.1, 0.3],
                    ]

# For ARDESPOT
bounds_ub = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(ping_first), ARDESPOT.FullyObservableValueUB(ValueIterationSolver(max_iterations=1000, include_Q=false)), check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [:default_action=>[ping_first,], 
                    :bounds=>[bounds_ub],
                    :K=>(k == 1 ? [100] : [100, 300]),
                    :lambda=>(k== 1 ? [0.0] : [0.0, 0.01, 0.1]),
                ]
ardespot_list_labels = [["PingFirst",], 
                        ["(PingFirst, MDP)"],
                        k == 1 ? [100] : [100, 300],
                        k== 1 ? [0.0] : [0.0, 0.01, 0.1],
                        ]

# For POMCPOW
value_estimator = FOValue(mdp)
pomcpow_list = [:default_action=>[ping_first,],
                    :estimate_value=>[value_estimator],
                    :tree_queries=>[200000,], 
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(17.0)],
                    :final_criterion=>[MaxTries()],
                    :max_depth=>[90],
                    :k_observation=>[6.0],
                    :alpha_observation=>[1/100.0],
                    :check_repeat_obs=>[false],
                    :enable_action_pw=>[false],
                ]
pomcpow_list_labels = [["PingFirst",],
                    ["MDP",],
                    [200000,], 
                    [1.0,],
                    [17.0],
                    ["MaxTries"],
                    [90],
                    [6.0],
                    [1/100.0],
                    [false],
                    [false],
                ]

# Solver list
solver_list = [
    DESPOTSolver=>ardespot_list, 
    POMCPOWSolver=>pomcpow_list,
    AdaOPSSolver=>adaops_list,
]
solver_list_labels = [
    ardespot_list_labels,
    pomcpow_list_labels,
    adaops_list_labels,
]

solver_labels = [
    "ARDESPOT",
    "POMCPOW",
    "AdaOPS",
]

episodes_per_domain = 1000
max_steps = 100
parallel_experiment(m,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=300,
                    experiment_label="SH1000",
                    full_factorial_design=true)