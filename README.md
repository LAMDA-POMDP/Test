# Test
A repo for testing script.
## Installation
```julia
Pkg> registry add git@github.com:JuliaPOMDP/Registry.git
Pkg> add git@github.com:LAMDA-POMDP/ParallelExperiment.jl.git git@github.com:LAMDA-POMDP/BS-DESPOT.jl.git git@github.com:LAMDA-POMDP/AdaOPS.jl.git git@github.com:LAMDA-POMDP/Roomba.jl.git git@github.com:LAMDA-POMDP/SubHunt.jl.git git@github.com:LAMDA-POMDP/VDPTag2.jl.git git@github.com:LAMDA-POMDP/LaserTag.jl.git git@github.com:JuliaPOMDP/RLInterface.jl.git git@github.com:JuliaPOMDP/POMDPGifs.jl.git git@github.com:LAMDA-POMDP/RockSample.jl.git git@github.com:LAMDA-POMDP/Multilane.jl.git
Pkg> instantiate
Pkg> precompile
```
## File Struture
### test folder
`test` folder contains all test scripts. The `Test.jl` is the main file, which will load all packages needed for testing. In `Test.jl`, you can change the `max_works` according to the specification of your machine and uncomment the domains you would like to test on.
Each `XXTest.jl` file contains the settings for the specific domain.
- BumperRoombaTest.jl: Roomba domain with a Bumper sensor
- LidarRoombaTest.jl: Roomba domain with a Lidar sensor
- LightDarkTest.jl: 1d Light Dark domain
- LTTest.jl: Laser Tag domain
- RSTest.jl: Rock Sample domain
- SHTest.jl: SubHunt domain (not ready yet, suffers from memory leakage)
- VDPTagTest.jl: VDPTag domain (will run, but very slow)
### analysis folder
Analysis files are used to visulize the algorithms so as to identify problems and tune hyperparameters.
### results folder
All results generated by `Test.jl` will be stored in `results` folder.
### scripts folder
Scripts folder contains scripts other than those for testing and analysis.
- MultilaneGIF.jl: For generating multilane gif.
- MultilaneTest.jl: For running different solvers in Multilane domain.
- ParticleFilterComparison.jl: For comparing the performance of different particle filters.
### figures folder
Figures folder contains `gif` and `svg` files generated by other scripts.