from collections import defaultdict
import hiive.mdptoolbox
from hiive.mdptoolbox import mdp
import hiive.mdptoolbox.example
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
N = 10
P, R = hiive.mdptoolbox.example.forest( S = N )
# fh = mdptoolbox.mdp.FiniteHorizon( P, R, 0.9, 3 )
# print( P )
# print( R )
# fh = hiive.mdptoolbox.mdp.PolicyIteration( P, R, 0.9 )
# fh.run()
# print( fh.run_stats[0].keys() )

STATS = [ 'Error', 'Time', 'Max V', 'Mean V', 'Iteration' ]
ALGOS = { 'PI': mdp.PolicyIteration,
          'VI': mdp.ValueIteration,
          'Q':  mdp.QLearning }

results = defaultdict( defaultdict )
for key, algo in ALGOS.items():
    fh = algo( P, R, 0.9 )
    if key == 'Q':
        fh = algo( P, R, 0.9, epsilon = 2, alpha = 0.4, n_iter = 100000 )
    fh.run()
    # print( fh.run_stats )
    for stat in STATS:
        results[ key ][ stat ] = [ fh.run_stats[ i ][ stat ] for i in range( len( fh.run_stats ) ) ]
    print( 'Algo %s: policy is %s' % ( key, fh.policy ) )

for stat in STATS:
    if stat == 'Iteration':
        continue
    for algo in ALGOS.keys():
        plt.plot( results[ algo ][ 'Iteration' ], results[ algo ][ stat ] )    
    plt.xlabel( 'Iterations' )
    plt.ylabel( stat )
    plt.title( 'Forest Management: N = %s' % N )
    plt.legend( list( ALGOS.keys() ) )
    plt.show()

plt.plot( results[ 'PI' ][ 'Time' ], results[ 'PI' ][ 'Error' ] )
plt.plot( results[ 'VI' ][ 'Time' ], results[ 'VI' ][ 'Error' ] )
plt.plot( results[ 'Q' ][ 'Time' ], results[ 'Q' ][ 'Mean V' ] )
plt.xlabel( 'Time' )
plt.ylabel( 'Error' )
plt.title( 'Forest Management: N = %s' % N )
plt.legend( [ 'PI', 'VI', 'Q' ] )
plt.show()

# gammas = []
# PI_deltas = []
# VI_deltas = []
# PI_iters = []
# VI_iters = []
# PI_means = []
# VI_means = []
# PI_rewards = []
# VI_rewards = []
# for gamma in [ 0.01 + 0.01 * i for i in range( 99 ) ]:
#     pi = mdp.PolicyIteration( P, R, gamma )
#     pi.run()
#     # print( pi.policy )
#     gammas.append( gamma )
#     PI_deltas.append( pi.run_stats[-1][ 'Error' ] )
#     PI_iters.append( pi.run_stats[-1][ 'Iteration' ] )
#     PI_means.append( pi.run_stats[-1][ 'Mean V' ] )
#     PI_rewards.append( np.max( [ run_stat[ 'Reward' ] for run_stat in pi.run_stats ] ) )

#     vi = mdp.ValueIteration( P, R, gamma )
#     vi.run()
#     VI_deltas.append( vi.run_stats[-1][ 'Error' ] )
#     VI_iters.append( vi.run_stats[-1][ 'Iteration' ] )
#     VI_means.append( vi.run_stats[-1][ 'Mean V' ] )
#     VI_rewards.append( np.max( [ run_stat[ 'Reward' ] for run_stat in vi.run_stats ] ) )
#     # print( fh.run_stats )
#     # print( fh.policy )

# # plt.plot( gammas, PI_means )
# # plt.plot( gammas, VI_means )
# # plt.xlabel( 'Gamma' )
# # plt.ylabel( 'Iterations Needed for Convergence' )
# # plt.title( 'Forest Management: N = 10' )
# # plt.legend( ['PI', 'VI'] )
# # plt.show()

# # plt.plot( gammas, PI_rewards )
# plt.plot( gammas, VI_rewards )
# plt.xlabel( 'Gamma' )
# plt.ylabel( 'Mean Reward' )
# plt.title( 'Forest Management: N = %s' % N )
# # plt.legend( ['PI', 'VI'] )
# plt.legend( ['VI'] )
# plt.show()

# # plt.plot( gammas, PI_iters )
# plt.plot( gammas, VI_iters )
# plt.xlabel( 'Gamma' )
# plt.ylabel( 'Iterations Needed for Convergence' )
# plt.title( 'Forest Management: N = %s' % N )
# # plt.legend( ['PI', 'VI'] )
# plt.legend( [ 'VI' ] )
# plt.show()

# plt.plot( gammas, PI_deltas )
# plt.xlabel( 'Gamma' )
# plt.ylabel( 'Error in Final Iteration' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( 'PI' )
# plt.show()

# plt.plot( gammas, VI_deltas )
# plt.xlabel( 'Gamma' )
# plt.ylabel( 'Error in Final Iteration' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( 'VI' )
# plt.show()


# for stat in STATS:
#     if stat == 'Iterations':
#         continue
#     plt.plot( results[ 'PI' ][ 'Iteration' ], results[ 'PI' ][ stat ] )
# plt.xlabel( 'Iterations' )
# plt.title( 'Policy Iteration' )
# plt.legend( STATS[:-1] )
# plt.show()




# print( fh.V )
# print( fh.policy )

# gammas = []
# q_deltas = []
# q_iters = []
# q_means = []
# q_rewards = []
# for gamma in [ 0.05 + 0.05 * i for i in range( 20 ) ]:
#     q = mdp.QLearning( P, R, gamma )
#     q.run()
#     # print( pi.policy )
#     gammas.append( gamma )
#     q_deltas.append( q.run_stats[-1][ 'Error' ] )
#     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
#     q_means.append( q.run_stats[-1][ 'Mean V' ] )
#     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )

# plt.plot( epsilons, q_iters )
# plt.xlabel( 'Alpha' )
# plt.ylabel( 'Iterations Needed for Convergence' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( 'Q' )
# plt.show()

# plt.plot( gammas, q_rewards )
# plt.xlabel( 'Gamma' )
# plt.ylabel( 'Mean Reward' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( ['Q'] )
# plt.show()



# alphas = []
# q_deltas = []
# q_iters = []
# q_means = []
# q_rewards = []
# for alpha in [ 0.02 + 0.02 * i for i in range( 10 ) ]:
#     q = mdp.QLearning( P, R, 0.9, alpha = alpha )
#     q.run()
#     # print( pi.policy )
#     alphas.append( alpha )
#     q_deltas.append( q.run_stats[-1][ 'Error' ] )
#     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
#     q_means.append( q.run_stats[-1][ 'Mean V' ] )
#     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )

# plt.plot( alphas, q_iters )
# plt.xlabel( 'Alpha' )
# plt.ylabel( 'Iterations Needed for Convergence' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( 'Q' )
# plt.show()

# plt.plot( alphas, q_rewards )
# plt.xlabel( 'Alpha' )
# plt.ylabel( 'Mean Reward' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( ['Q'] )
# plt.show()

# epsilons = []
# q_deltas = []
# q_iters = []
# q_means = []
# q_rewards = []
# for epsilon in [ 0.5 + 0.1 * i for i in range( 15 ) ]:
#     q = mdp.QLearning( P, R, 0.9, epsilon = epsilon )
#     q.run()
#     # print( pi.policy )
#     epsilons.append( epsilon )
#     q_deltas.append( q.run_stats[-1][ 'Error' ] )
#     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
#     q_means.append( q.run_stats[-1][ 'Mean V' ] )
#     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )

# plt.plot( epsilons, q_iters )
# plt.xlabel( 'Alpha' )
# plt.ylabel( 'Iterations Needed for Convergence' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( 'Q' )
# plt.show()

# plt.plot( epsilons, q_rewards )
# plt.xlabel( 'Epsilon' )
# plt.ylabel( 'Mean Reward' )
# plt.title( 'Forest Management: N = %s' % N )
# plt.legend( ['Q'] )
# plt.show()