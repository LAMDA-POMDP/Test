using AA228FinalProject
using POMDPs
using POMDPSimulators
using MCVI

max_speed = 2.0
speed_interval = 2.0
max_turn_rate = 1.0
turn_rate_interval = 1.0
action_space = vec([RoombaAct(v, om) for v in 0:speed_interval:max_speed, om in -max_turn_rate:turn_rate_interval:max_turn_rate])
sensor = Lidar() # or Bumper() for the bumper version of the environment
config = 3 # 1,2, or 3
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config, aspace=action_space, v_max=max_speed))

# Belief updater
num_particles = 5000
v_noise_coefficient = 2.0
om_noise_coefficient = 0.5
belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient)

struct LidarRoombaTrivialBound end
function MCVI.lower_bound(::LidarRoombaTrivialBound, m::RoombaPOMDP, s::RoombaState) 
    mdp = AA228FinalProject.mdp(m)
    (mdp.contact_pen + mdp.time_pen) / (1 - discount(mdp))
end
MCVI.upper_bound(::LidarRoombaTrivialBound, m::RoombaPOMDP, s::RoombaState) = AA228FinalProject.mdp(m).goal_reward

solver = MCVISolver(RolloutSimulator(max_steps=100), nothing, 1, 3000, 8, 500, 1000, 5000, 50, LidarRoombaTrivialBound(), LidarRoombaTrivialBound())
policy = solve(solver, m)
