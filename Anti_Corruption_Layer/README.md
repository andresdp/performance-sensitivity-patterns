# Anti-corruption Layer

This folder contains models, scripts, data, and results of the [Anti-corruption Layer](https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer) design pattern.


## Print figures in the paper
Please, follow these steps to print the same figures presented in the paper for this design pattern:
1. Start a Jupyter notebook using <tt>jupyter notebook</tt>.
2. Open <tt>analysis.ipynb</tt>.
3. Execute all cells in that file.


## Reproduce results in the paper
To reproduce results presented in the paper, users can run new simulation. In this case, they should note that slightly different values may be observed due to the stochastic simulation. Please, follow these steps:
1. Run <tt>python3 runSim.py</tt>.
2. Start a Jupyter notebook using <tt>jupyter notebook</tt>.
3. Open <tt>analysis.ipynb</tt>.
4. Execute all cells in that file.


## Run provided models with different input parameters
Users can also tune input parameters of provided models to study different applications designed with the same patter.
Please, follow these steps to study your own applications with our model:
1. Open <tt>runSim.py</tt>
2. Edit lines 55--69 with your input parameters (see the [Parameter Table](#parameter-table) for detailed explanation of available parameters) and save the file.
3. Run <tt>python3 runSim.py</tt>.
4. Start a Jupyter notebook using <tt>jupyter notebook</tt>.
5. Open <tt>analysis.ipynb</tt>.
6. Execute all cells in that file.

Note that all input parameters (except <tt>OUTFILE</tt>) are defined as lists to facilitate the execution of multiple simulation. If lists are not used, the tool will stop working. Use a list with *length = 1* to consider a single value for a parameter; otherwise, use a list with *length = N* to study your system with *N* different values for the considered parameter. 

The tool *combines* all provided input parameters and executes as many simulations as the total number of combinations of input parameters.


### Parameter Table

| Parameter | Description |
| --- | --- |
| OUTFILE | Name of the file where simulation results will be stored |
| N\_TOT | Total number of requests in the system |
| N\_A | Number of type A requests in the system |
| Z\_A | Think time of type A requests |
| Z\_B | Think time of type B requests |
| C\_SS1 | Number of CPUs allocated to sub-suystem 1 |
| p\_A\_SS1toACL | Probability that a type A request goes is routed to the anti-corruption layer after visiting sub-system 1 |
| p\_B\_SS1toACL | Probability that a type B request goes is routed to the anti-corruption layer after visiting sub-system 1 |
| S\_A\_SS1 | Service time of type A requests at sub-system 1 |
| S\_B\_SS1 | Service time of type B requests at sub-system 1 |
| S\_A\_ACL | Service time of type A requests at the anti-corruption layer |
| S\_B\_ACL | Service time of type B requests at the anti-corruption layer |
| S\_A\_SS2 | Service time of type A requests at sub-system 2 |
| S\_B\_SS2 | Service time of type B requests at sub-system 2 |
