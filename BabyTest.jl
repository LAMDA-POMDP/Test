@everywhere using POMDPModels

@everywhere convert(s::Bool, pomdp::BabyPOMDP) = Float64[s]
grid = StateGrid(convert, [1.0])
# Type stability
pomdp = BabyPOMDP()
mdp = solve(ValueIterationSolver(max_iterations=1000, include_Q=false), pomdp)
sarsop = solve(SARSOPSolver(fast=true, max_iterations=1000), pomdp)
spl_bounds = IndependentBounds(SemiPORollout(FeedWhenCrying()), mdp)
pl_bounds = IndependentBounds(PORollout(FeedWhenCrying(), PreviousObservationUpdater()), mdp)
solver = AdaOPSSolver(bounds=bds,
                      rng=MersenneTwister(4),
                      grid=grid,
                      zeta=0.04,
                      delta=0.04,
                      xi=0.1,
                      ESS=true,
                      m_min=1.0,
                      tree_in_info=true
                     )
p = solve(solver, pomdp)