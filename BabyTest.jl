@everywhere using POMDPModels

@everywhere convert(s::Bool, pomdp::BabyPOMDP) = Float64[s]
grid = StateGrid(convert, [1.0])
# Type stability
pomdp = BabyPOMDP()
# qmdp = solve(QMDPSolver(max_iterations=1000), pomdp)
# sarsop = solve(SARSOPSolver(fast=true, timeout=1000), pomdp)
# @show value(sarsop, initialstate(pomdp))
# @show value(qmdp, initialstate(pomdp))

solver = AdaOPSSolver(bounds=pl_bounds,
                      grid=grid,
                      zeta=0.04,
                      delta=0.04,
                      xi=0.1,
                      m_init=30.0,
                      sigma=10.0,
                      tree_in_info=true
                     )
bounds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), 0.0)
adaops_list = [
            :default_action=>[FeedWhenCrying(),],
            :bounds=>[bounds,],
            :delta=>[0.02, 0.04, 0.06],
            :grid=>[grid],
            :m_init=>[30, 50, 100],
            :sigma=>[2.0, 3.0, 5.0],
            :zeta=>[0.02, 0.03, 0.04, 0.05],
            :bounds_warnings=>[false],
            ]
adaops_list_labels = [
                    ["FeedWhenCrying",],
                    ["(FeedWhenCrying, 0.0)"],
                    [0.02, 0.04, 0.06],
                    [grid],
                    [30, 50, 100],
                    [2.0, 3.0, 5.0],
                    [0.02, 0.03, 0.04, 0.05],
                    [false],
                    ]
# For PLDESPOT
bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(FeedWhenCrying()), 0.0)

ardespot_list = [
    :default_action=>[FeedWhenCrying()],
    :bounds=>[bounds],
    :K=>[100, 300, 500, 1000],
    :beta=>[0.0, 0.01, 0.1],
    ]

ardespot_list_labels = [
    ["FeedWhenCrying"],
    ["(FeedWhenCrying, 0.0)"],
    [100, 300, 500, 1000],
    [0.0, 0.01, 0.1],
    ]

# For POMCPOW
pomcpow_list = [:default_action=>[FeedWhenCrying()],
                    :estimate_value=>[PORollout(FeedWhenCrying(), PreviousObservationUpdater())],
                    :tree_queries=>[200000,], 
                    :max_time=>[1.0,],
                    :criterion=>[MaxUCB(1.), MaxUCB(10.), MaxUCB(100.), MaxUCB(1000.)]]
pomcpow_list_labels = [["FeedWhenCrying",],
                    ["FeedWhenCrying",],
                    [200000,], 
                    [1.0, 10.0, 100.0, 1000.0],
                    ]

# Solver list
solver_list = [
    PL_DESPOTSolver=>ardespot_list, 
    POMCPOWSolver=>pomcpow_list,
    AdaOPSSolver=>adaops_list,
]
solver_list_labels = [
    ardespot_list_labels,
    pomcpow_list_labels,
    adaops_list_labels,
]

solver_labels = [
    "PLDESPOT",
    "POMCPOW",
    "AdaOPS",
]

episodes_per_domain = 1000
max_steps = 50
parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    max_queue_length=320,
                    domain_queue_length=5,
                    experiment_label="CryingBaby",
                    full_factorial_design=true)