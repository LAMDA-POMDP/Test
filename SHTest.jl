@everywhere using SubHunt

# Low passive std may help agent identify useful information
info_gather_pomdp = SubHuntPOMDP(passive_std=1.0, p_aware_kill=0.05)
pomdp = SubHuntPOMDP()

# Choose default domain
m = info_gather_pomdp
pomdp_gen() = SubHuntPOMDP(passive_std=1.0, p_aware_kill=0.05)
# m = pomdp
qmdp= QMDPSolver(max_iterations=1000)
mdp = ValueIterationSolver(max_iterations=1000)
random = RandomSolver()
random_policy = solve(random, m)

@everywhere convert(s::SubState, pomdp::SubHuntPOMDP) = SVector{3, Float64}(s.target..., s.goal)
grid = StateGrid(convert, range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=pomdp.size, length=5)[2:end],
                            range(1, stop=4, length=4)[2:end]
                            )
@everywhere POMDPs.action(p::PingFirst, b::SubHunt.SubHuntInitDist) = SubHunt.PING
@everywhere POMDPs.action(p::AlphaVectorPolicy, s::SubState) = action(p, ParticleCollection([s]))

# splpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(qmdp), POValue(qmdp))
# splpu_bounds = AdaOPS.IndependentBounds(SemiPORollout(ping_first_qmdp), POValue(qmdp))
pu_bounds = AdaOPS.IndependentBounds(FORollout(random), POValue(qmdp))

adaops_list = [:default_action=>[random_policy],
# adaops_list = [:default_action=>[ping_first_qmdp,],
            :bounds=>[pu_bounds],
            :delta=>[0.3, 0.6, 1.0],
            :grid=>[grid],
            :m_init=>[5, 10, 20],
            :sigma=>[2.0, 5.0, 8.0],
            :zeta=>[0.03, 0.1, 0.3],
            :bounds_warnings=>[false]
            ]
adaops_list_labels = [["Random",],
                    ["(RandomRollout, QMDP)"],
                    [0.3, 0.6, 1.0],
                    ["FullGrid"],
                    [5, 10, 20],
                    [2, 5, 8],
                    [0.03, 0.1, 0.3],
                    [false]
                    ]

# For ARDESPOT
bounds_ub = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(random), qmdp, check_terminal=true, consistency_fix_thresh=1e-5)

ardespot_list = [:default_action=>[random_policy], 
                    :bounds=>[bounds_ub],
                    :K=>[200, 300, 400],
                    :lambda=>[0.0, 0.01, 0.1],
                    :bounds_warnings=>[false]
                ]
ardespot_list_labels = [["Random",], 
                        ["(RandomRollout, QMDP)"],
                        [200, 300, 400],
                        [0.0, 0.01, 0.1],
                        [false]
                        ]

# For POMCPOW
pomcpow_list = [:default_action=>[random_policy],
                    :estimate_value=>[FOValue(mdp)],
                    :tree_queries=>[200000,], 
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(10.0), MaxUCB(17.0), MaxUCB(25.0)],
                    :final_criterion=>[MaxTries()],
                    :max_depth=>[90],
                    :k_observation=>[3.0, 6.0, 9.0],
                    :alpha_observation=>[1/50, 1/100.0, 1/200.0],
                    :check_repeat_obs=>[false],
                    :enable_action_pw=>[false],
                ]
pomcpow_list_labels = [["Random",],
                    ["MDP",],
                    [200000,], 
                    [1.0,],
                    [10.0, 17.0, 25.0],
                    ["MaxTries"],
                    [90],
                    [3.0, 6.0, 9.0],
                    [1/50, 1/100.0, 1/200.0],
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

episodes_per_domain = 10
max_steps = 100
parallel_experiment(pomdp_gen,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=100,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    domain_queue_length=1,
                    max_queue_length=256,
                    experiment_label="SH1000",
                    full_factorial_design=true)
