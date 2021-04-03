using Pkg
Pkg.activate(".")

using SARSOP
using POMDPs
using POMDPSimulators
using RockSample


for i in 1:10
    m = RockSamplePOMDP(7, 8)
    policy = solve(SARSOPSolver(), m)
    r = 0.0
    for i in 1:100
        sim = RolloutSimulator(max_steps=100)
        r += simulate(sim, m, policy)
    end
    @show r/100
end