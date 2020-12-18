@everywhere using VDPTag2

cpomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 1.8)))
pomdp = ADiscreteVDPTagPOMDP(cpomdp=cpomdp)
manage_uncertainty = ManageUncertainty(pomdp, 2.0)
to_next_ml = ToNextML(pomdp)

@everywhere POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::TagState, m) = SVector{4,Float64}(s.agent..., s.target...)
@everywhere POMDPs.convert_s(::Type{TagState}, v::AbstractVector{Float64}, m) = TagState(v[1:2], v[3:4])

VERTICES_PER_AXIS = 10 # Controls the resolutions along the grid axis
grid = RectangleGrid(range(-4, stop=4, length=VERTICES_PER_AXIS),
                    range(-4, stop=4, length=VERTICES_PER_AXIS),
                    range(-4, stop=4, length=VERTICES_PER_AXIS),
                    range(-4, stop=4, length=VERTICES_PER_AXIS)) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

approx_solver = LocalApproximationValueIterationSolver(interp,
                                                        verbose=true,
                                                        max_iterations=1000,
                                                        is_mdp_generative=true,
                                                        n_generative_samples=1000)
approx_mdp = solve(approx_solver, UnderlyingMDP(pomdp))

# For AdaOPS
@everywhere convert(s::TagState, pomdp) = s.target
grid = StateGrid(convert,
                range(-4, stop=4, length=5)[2:end-1],
                range(-4, stop=4, length=5)[2:end-1])
flfu_bounds = AdaOPS.IndependentBounds(FORollout(to_next_ml), FOValue(approx_mdp), check_terminal=true)
splfu_bounds = AdaOPS.IndependentBounds(SemiPORollout(manage_uncertainty), FOValue(approx_mdp), check_terminal=true)
adaops_list = [:default_action=>[manage_uncertainty,], 
                    :bounds=>[flfu_bounds, splfu_bounds],
                    :delta=>[0.1, 0.3],
                    :grid=>[nothing, grid],
                    :m_init=>[30, 50],
                    :zeta=>[0.1, 0.3],
                    :xi=>[0.1, 0.3, 0.95]]

adaops_list_labels = [["ManageUncertainty",], 
                    ["(FO_ToNextML, MDP)", "(SemiPO_ManageUncertainty, MDP)"],
                    [0.1, 0.3],
                    ["NullGrid", "FullGrid"],
                    [30, 50],
                    [0.1, 0.3],
                    [0.1, 0.3, 0.95]]
# ARDESPOT
fo_bounds = ARDESPOT.IndependentBounds(ARDESPOT.DefaultPolicyLB(manage_uncertainty), ARDESPOT.FullyObservableValueUB(approx_mdp), check_terminal=true)
ardespot_list = [:default_action=>[manage_uncertainty], 
                :bounds=>[fo_bounds,],
                :lambda=>[0.1,],
                :K=>[300],
                ]
ardespot_list_labels = [["ManageUncertainty",], 
                ["(ManageUncertainty, MDP)",],
                [0.1,],
                [300],
                ]

# For POMCPOW
running_estimator = FORollout(to_next_ml)
mdp_estimator = FOValue(approx_mdp)
pomcpow_list = [:default_action=>[running,], 
                :estimate_value=>[running_estimator, mdp_estimator],
                :tree_queries=>[100000,], 
                :max_time=>[1.0,], 
                :criterion=>[MaxUCB(1000.),]]
pomcpow_list_labels = [["ToNextML",], 
                        ["ToNextMLRollout", "MDPValue"],
                        [100000,], 
                        [1.0,], 
                        ["UCB 1000"]]

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

                
episodes_per_domain = 300
max_steps = 100

parallel_experiment(pomdp,
                    episodes_per_domain,
                    max_steps,
                    solver_list,
                    num_of_domains=3,
                    solver_labels=solver_labels,
                    solver_list_labels=solver_list_labels,
                    max_queue_length=300,
                    belief_updater=belief_updater,
                    experiment_label="Roomba3*300",
                    full_factorial_design=true)