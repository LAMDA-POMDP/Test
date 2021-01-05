@everywhere using VDPTag2

cpomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))
pomdp = ADiscreteVDPTagPOMDP(cpomdp=cpomdp)
dpomdp = AODiscreteVDPTagPOMDP(cpomdp, 25, 0.2)
manage_uncertainty = ManageUncertainty(pomdp, 0.01)
to_next_ml = ToNextML(pomdp)

@everywhere struct RootToNextMLFirst
    rng::AbstractRNG
end

@everywhere function MCTS.next_action(gen::RootToNextMLFirst, p::VDPTagPOMDP, b, node)
    if isroot(node) && n_children(node) < 1
        target_sum=MVector(0.0, 0.0)
        agent_sum=MVector(0.0, 0.0)
        for s in particles(b::ParticleCollection)
            target_sum += s.target
            agent_sum += s.agent
        end
        next = VDPTag.next_ml_target(mdp(p), target_sum/n_particles(b))
        diff = next-agent_sum/n_particles(b)
        return TagAction(false, atan2(diff[2], diff[1]))
    else
        return rand(gen.rng, actions(p))
    end
end

@everywhere function VDPUpper(pomdp, b)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0
    else
        return mdp(cproblem(pomdp)).tag_reward
    end
end
rng = Random.GLOBAL_RNG

# For AdaOPS
@everywhere convert(s::TagState, pomdp) = s.target
grid = StateGrid(convert,
                range(-4, stop=4, length=5)[2:end-1],
                range(-4, stop=4, length=5)[2:end-1])
random_estimator = FORollout(RandomSolver())
flfu_bounds = AdaOPS.IndependentBounds(random_estimator, VDPUpper, check_terminal=true, consistency_fix_thresh=1e-5)
adaops_list = [:default_action=>[manage_uncertainty,], 
                    :bounds=>[flfu_bounds],
                    :delta=>[0.8, 1.2, 2.0],
                    :grid=>[nothing, grid],
                    :m_init=>[10, 30, 50],
                    :sigma=>[3, 10],
                    :zeta=>[0.1, 0.3, 0.5],
]

adaops_list_labels = [["ManageUncertainty",], 
                    ["(RandomRollout, VDPUpper)"],
                    [0.8, 1.2, 2.0],
                    ["NullGrid", "FullGrid"],
                    [10, 30, 50],
                    [3, 10],
                    [0.1, 0.3, 0.5],
]
# ARDESPOT
fo_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(manage_uncertainty), VDPUpper, check_terminal=true, consistency_fix_thresh=1e-5)
ardespot_list = [:default_action=>[manage_uncertainty], 
                :bounds=>[fo_bounds,],
                :lambda=>[0.0, 0.01,],
                :K=>[200, 300],
                :random_source=>[ARDESPOT.MemorizingSource(500, 10, rng, min_reserve=8)],
                ]
ardespot_list_labels = [["ManageUncertainty",], 
                ["(ManageUncertainty, MDP)",],
                [0.0, 0.01,],
                [200, 300],
                ["RandomSource"],
                ]

# For POMCPOW
random_estimator = FORollout(RandomSolver())
pomcpow_list = [ 
                :default_action=>[manage_uncertainty],
                :estimate_value=>[random_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(110.0)],
                :final_criterion=>[MaxQ()],
                :max_depth=>[90],
                :k_action=>[30.0],
                :alpha_action=>[1/30],
                :k_observation=>[5.0],
                :alpha_observation=>[1/100],
                :next_action=>[RootToNextMLFirst(rng)],
                :check_repeat_obs=>[false],
                :check_repeat_act=>[false],
                ]
pomcpow_list_labels = [ 
                        ["ManageUncertainty"],
                        ["RandomRollout"],
                        [100000,], 
                        [1.0,], 
                        [110.0],
                        ["MaxQ"],
                        [90],
                        [30.0],
                        [1/30],
                        [5.0],
                        [1/100],
                        ["RootToNextMLFirst"],
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

                
episodes_per_domain = 100
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=1,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=300,
                    belief_updater=(m)->BasicParticleFilter(m, POMDPResampler(30000), 30000),
                    experiment_label="VDPTag*100",
                    full_factorial_design=true)

# # Solver list
# solver_list = [
#                 # AdaOPSSolver=>adaops_list, 
#                 DESPOTSolver=>ardespot_list,
#                 # POMCPOWSolver=>pomcpow_list,
#                 ]
# solver_list_labels = [
#                     # adaops_list_labels, 
#                     ardespot_list_labels,
#                     # pomcpow_list_labels,
#                     ]
# solver_labels = [
#                 # "ADAOPS",
#                 "ARDESPOT",
#                 # "POMCPOW",
#                 ]

                
# episodes_per_domain = 100
# max_steps = 100

# parallel_experiment(dpomdp,
#                     episodes_per_domain,
#                     max_steps,
#                     solver_list,
#                     num_of_domains=1,
#                     solver_labels=solver_labels,
#                     solver_list_labels=solver_list_labels,
#                     max_queue_length=300,
#                     belief_updater=belief_updater,
#                     experiment_label="VDPTag*100",
#                     full_factorial_design=true)

# Solver list
solver_list = [
                # AdaOPSSolver=>adaops_list, 
                # DESPOTSolver=>ardespot_list,
                POMCPOWSolver=>pomcpow_list,
                ]
solver_list_labels = [
                    # adaops_list_labels, 
                    # ardespot_list_labels,
                    pomcpow_list_labels,
                    ]
solver_labels = [
                # "ADAOPS",
                # "ARDESPOT",
                "POMCPOW",
                ]

                
episodes_per_domain = 100
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
                    experiment_label="VDPTag*100",
                    full_factorial_design=true)