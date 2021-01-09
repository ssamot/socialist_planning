# open loop in natura economic planning
An initial implementation of "open loop in natura economic planning". Note that
this is not your standard "data - train - validate - test" machine learning code, the
topic of the paper is very different, and the purpose of the experiments is to showcase
feasibility, rather then benchmark vs a previous method (which does not exist). 

All the generated data for the plots is in `./plot_data`. All data required for the 
village experiment is in `./data/`. The random generated matrices are missing, but
(a) they can be generated using `./generate_matrices.py` (b) they are 22GB. 


`./selvaria.py` executes the experiments and generates the experiment for
optimising the economy of a small alien village.

`./online_time_complexity.py` optimises for one tick for various random 
economies of different sizes. All the generated files are around 22GB, so they are 
not included (but you can generate them). 

`./coeffs.py` has the coefficients for the production units of the village economy. 
The corresponding consumption data is in `./data`. If you just run the file it will 
create the coefficient plots for those three production units (i.e. factories). 

`./utils.py` include various helper functions. 

`./generate_matrices.py` includes the code to generate random sparse matrices. 

`./plot_selvaria.py` includes the code for investment plots, as presented in the paper. 

`./plot_time.py` plots the (empirical) time complexity of solving for matrices 
of certain size. 

`./metrics.py` includes the code for humanity and other metrics. 

