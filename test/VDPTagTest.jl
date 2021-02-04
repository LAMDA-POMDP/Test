@everywhere using VDPTag2

cpomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))
pomdp = ADiscreteVDPTagPOMDP(cpomdp=cpomdp)
dpomdp = AODiscreteVDPTagPOMDP(cpomdp, 10, 5.0)
manage_uncertainty = ManageUncertainty(pomdp, 0.01)
to_next_ml = ToNextML(pomdp)

@everywhere function VDPUpper(pomdp, b)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0
    else
        return mdp(pomdp).tag_reward
    end
end
rng = Random.GLOBAL_RNG

# For AdaOPS
@everywhere Base.convert(::Type{SVector{2,Float64}}, s::TagState) = s.target
grid = StateGrid(range(-4, stop=4, length=5)[2:end-1],
                range(-4, stop=4, length=5)[2:end-1])

random_estimator = FORollout(RandomSolver())
bounds = AdaOPS.IndependentBounds(random_estimator, mdp(pomdp).tag_reward, check_terminal=true, consistency_fix_thresh=1e-5)
spl_bounds = AdaOPS.IndependentBounds(SemiPORollout(manage_uncertainty), mdp(pomdp).tag_reward, check_terminal=true, consistency_fix_thresh=1e-5)
fl_bounds = AdaOPS.IndependentBounds(FORollout(to_next_ml), mdp(pomdp).tag_reward, check_terminal=true, consistency_fix_thresh=1e-5)
adaops_list = [
                :default_action=>[manage_uncertainty,], 
                :bounds=>[bounds, fl_bounds],
                :delta=>[0.5, 1.0],
                :grid=>[grid],
                :m_init=>[10, 20],
                :sigma=>[2.0, 3.0, 4.0],
                :zeta=>[0.03, 0.1, 0.3],
]

adaops_list_labels = [
                    ["ManageUncertainty",], 
                    ["(RandomRollout, $(mdp(pomdp).tag_reward)", "(FO_ToNextML, $(mdp(pomdp).tag_reward)"],
                    [0.5, 1.0],
                    ["FullGrid"],
                    [10, 20],
                    [2.0, 3.0, 4.0],
                    [0.03, 0.1, 0.3],
]
# ARDESPOT
fo_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(manage_uncertainty), VDPUpper, check_terminal=true, consistency_fix_thresh=1e-5)
ardespot_list = [:default_action=>[manage_uncertainty], 
                :bounds=>[fo_bounds,],
                :lambda=>[0.0, 0.01, 0.1],
                :K=>[100, 200, 300],
                ]
ardespot_list_labels = [["ManageUncertainty",], 
                ["(ManageUncertainty, VDPUpper)",],
                [0.0, 0.01, 0.1],
                [100, 200, 300],
                ]

# For POMCPOW
random_estimator = FORollout(RandomSolver())
to_next_ml_estimator = FORollout(to_next_ml)
pomcpow_list = [ 
                #:default_action=>[manage_uncertainty],
                :estimate_value=>[random_estimator, to_next_ml_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(110.0)],
                :final_criterion=>[MaxQ()],
                :max_depth=>[90],
                :k_action=>[30.0],
                :alpha_action=>[1/30],
                :k_observation=>[3.0, 5.0, 7.0, 10.0],
                :alpha_observation=>[1/200, 1/100, 1/50, 1/25],
                :next_action=>[NextMLFirst(rng)],
                :check_repeat_obs=>[false],
                :check_repeat_act=>[false],
                ]
pomcpow_list_labels = [ 
                        #["ManageUncertainty"],
                        ["RandomRollout", "to_next_ml"],
                        [100000,], 
                        [1.0,], 
                        [110.0],
                        ["MaxQ"],
                        [90],
                        [30.0],
                        [1/30],
                        [3.0, 5.0, 7.0, 10.0],
                        [1/200, 1/100, 1/50, 1/25],
                        ["NextMLFirst"],
                        [false],
                        [false],
                        ]

# Solver list
solver_list = [
                AdaOPSSolver=>adaops_list, 
                DESPOTSolver=>ardespot_list,
                POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    adaops_list_labels, 
                    ardespot_list_labels,
                    pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                "ARDESPOT",
                "POMCPOW",
                ]

                
episodes_per_domain = 1000
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=960,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    experiment_label="VDPTag1000",
                    full_factorial_design=true)
# # Solver list
# solver_list = [
#                  AdaOPSSolver=>adaops_list, 
#                  DESPOTSolver=>ardespot_list,
#                  # POMCPOWSolver=>pomcpow_list,
#                  ]
# solver_list_labels = [
#                    adaops_list_labels, 
#                    ardespot_list_labels,
#                    # pomcpow_list_labels,
#                     ]
# solver_labels = [
#                 "ADAOPS",
#                 "ARDESPOT",
#                 # "POMCPOW",
#                 ]

                
# episodes_per_domain = 300
# max_steps = 100

# parallel_experiment(dpomdp,
#                     episodes_per_domain,
#                     max_steps,
#                     solver_list,
#                     num_of_domains=1,
#                     solver_labels=solver_labels,
#                     solver_list_labels=solver_list_labels,
#                     max_queue_length=900,
#                     belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
#                     experiment_label="VDPTag*100_addis",
#                     full_factorial_design=true)

#= Solver list
solver_list = [
                AdaOPSSolver=>adaops_list, 
                # DESPOTSolver=>ardespot_list,
                #POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    adaops_list_labels, 
                    # ardespot_list_labels,
                    # pomcpow_list_labels,
                    ]
solver_labels = [
                "ADAOPS",
                # "ARDESPOT",
                #"POMCPOW",
                ]

                
episodes_per_domain = 300
max_steps = 100

parallel_experiment(cpomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=300,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    experiment_label="VDPTag*100_con_para1",
                    full_factorial_design=true)=#
