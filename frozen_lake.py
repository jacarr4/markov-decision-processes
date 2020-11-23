import hiive.mdptoolbox
from hiive.mdptoolbox import mdp
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

def run(func):
    func()
    return func

N = 400

# THIS EXAMPLE TAKEN FROM CMARON'S GITHUB: https://github.com/cmaron/CS-7641-assignments/tree/master/assignment4
# Display code adapted from there as well. 
# grid_world_example code taken from stackoverflow.
GRID_DESC = [ "SFFFFFFHHHFFFFFFFFFF",
              "FFFFFFFFFFFFFFFFHHFF",
              "FFFHFFFFFFFHHFFFFFFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFFFFHFFFFFFFHHFF",
              "FFFFFHFFFFHHFFFFHHFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFFFFFFFHHHHHHHFF",
              "HHHHFHFFFFFFFFFFHHFF",
              "FFFFFHFFFFHHHFFFHHFF",
              "FFFFFFFFFFFFFFFFHHFF",
              "FFFFFHFFFFFFHFFFHHFF",
              "FFFFFHFFFFFFFFFFHHFF",
              "FFFFFFFFFFFHFFFFFFFF",
              "FHHFFFHFFFFHFFFFFHFF",
              "FHHFHFHFFFFFFFFFFFFF",
              "FFFHFFFFFHFFFFHHFHFG"
]

# N = 64
# GRID_DESC = [ "SFFFFFFF",
#               "FFFFFFFF",
#               "FFFHFFFF",
#               "FFFFFHFF",
#               "FFFHFFFF",
#               "FHHFFFHF",
#               "FHFFHFHF",
#               "FFFHFFFG"
# ]

def parse_grid( grid ):
    size = ( len( grid ), len( grid[0] ) )
    red_cells = []
    for y in range( len( grid ) ):
        for x in range( len( grid[ 0 ] ) ):
            if grid[ y ][ x ] == 'H':
                red_cells.append( ( y, x ) )
            elif grid[ y ][ x ] == 'S':
                start_loc = ( y, x )
            elif grid[ y ][ x ] == 'G':
                green_cell_loc = ( y, x )
    return size, red_cells, start_loc, green_cell_loc

def grid_world_example(grid_size=(3, 4),
                       black_cells=[],
                       white_cell_reward=-0.1,
                       green_cell_loc=(0,3),
                       red_cells = [],
                       green_cell_reward=10000.0,
                       red_cell_reward=-100.0,
                       action_lrfb_prob=(.1, .1, .8, 0.),
                       start_loc=(0, 0)
                      ):
    # Constructing a grid for visualization
    
    grid = [[0]*grid_size[1] for _ in range(grid_size[0])]
   
    for black_cell in black_cells:
        grid[black_cell[0]][black_cell[1]] = -1
    
    # for red_cell in red_cells:
    #     grid[red_cell[0]][red_cell[1]] = -20
    
    grid[start_loc[0]][start_loc[1]] = 10
    grid[green_cell_loc[0]][green_cell_loc[1]] = 20
    # grid[red_cell_loc[0]][red_cell_loc[1]] = -20

    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    @run
    def fill_in_probs():
        # helpers
        to_2d = lambda x: np.unravel_index(x, grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, grid_size)

        def hit_wall(cell):
            if cell in black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}

        # work in terms of the 2d grid representation

        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell == green_cell_loc:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = green_cell_reward

            elif cell in red_cells:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                R[to_1d(cell), a_index] = white_cell_reward

            else:
                P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                R[to_1d(cell), a_index] = white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(grid_size):
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['up'])

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['down'])

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, action['left'])
                                # right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, action['right'])

    return P, R, grid

def policy_to_dir(policy):
    if policy == 0: # up
        return 0, 20
    elif policy == 1: # left
        return -20, 0
    elif policy == 2: # down
        return 0 ,- 20
    elif policy == 3: # right
        return 20, 0

COLORS = {
    'S': 'green',
    'F': 'skyblue',
    'H': 'black',
    'G': 'gold'
}

DIRECTIONS = {
    0: '⬆',
    3: '➡',
    1: '⬇',
    2: '⬅'
}

def display_grid(grid, values, policy):

    # create discrete colormap
    policy = np.array(policy).reshape(len(grid), len(grid[0]))
    values = np.array(values).reshape(len(grid), len(grid[0]))

    data = grid
    print(grid)
    cmap = colors.ListedColormap(['red', 'black', 'white','blue', 'green'])
    bounds = [-30, -15, -5, 5, 15, 25]
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)

    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'small'
    for y in range( len( policy ) ):
        for x in range( len( policy[0] ) ):
            # print( '(x,y)=(%s,%s)' % (x,y) )
            newY = len( grid ) - y - 1
            p = plt.Rectangle([x, newY], 1, 1)
            # print( GRID_DESC[ y ][ x ] )
            p.set_facecolor( COLORS[ GRID_DESC[ y ][ x ] ] )
            ax.add_patch(p)
            text = ax.text(x+0.5, y+0.5, DIRECTIONS[policy[newY, x]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])
    # ax.imshow(data, cmap=cmap, norm=norm)

    # # draw gridlines
    # ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    # ax.set_xticks(np.arange(0, len(grid[0]), 1))
    # ax.set_yticks(np.arange(0, len(grid), 1))
    # # quiver([X, Y], U, V, [C], **kw)
    # # adding the policy and value overlays
    # for x in range(len(policy)):
    #     for y in range(len(policy[0])):
    #         u, v = policy_to_dir(policy[x][y])
    #         plt.quiver(y, x, u, v, color='lime')
    # # for i, ((x,y),) in enumerate(zip(xy)):
    # #     plt.text(x,y,i, ha="center", va="center")


    plt.axis('off')
    plt.title( 'Q-Learning: Frozen Lake, N = %s' % N )
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.show()

def main():
    size, red_cells, start_loc, green_cell_loc = parse_grid( GRID_DESC )
    P, R, grid = grid_world_example(grid_size = size,
                       black_cells = [],
                       white_cell_reward=-0.1,
                       green_cell_loc = green_cell_loc,
                    #    red_cell_loc=(1,3),
                       red_cells = red_cells,
                       green_cell_reward = 400.0,
                       red_cell_reward = -10.0,
                       action_lrfb_prob=(.05, .05, .9, 0.),
                       start_loc = start_loc
                      )

    
    # uncomment these to graph
    # q = mdp.QLearning(P, R, 0.9)
    # q.run()
    # display_grid(grid, q.V, q.policy)

    # exit( 0 )

    # gammas = []
    # q_deltas = []
    # q_iters = []
    # q_means = []
    # q_rewards = []
    # for gamma in [ 0.05 + 0.05 * i for i in range( 20 ) ]:
    #     q = mdp.QLearning( P, R, gamma )
    #     q.run()
    #     gammas.append( gamma )
    #     q_deltas.append( q.run_stats[-1][ 'Error' ] )
    #     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
    #     q_means.append( q.run_stats[-1][ 'Mean V' ] )
    #     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )
    #     print( q.run_stats )

    # plt.plot( gammas, q_rewards )
    # plt.xlabel( 'Gamma' )
    # plt.ylabel( 'Mean Reward' )
    # plt.title( 'Frozen Lake: N = %s' % N )
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
    #     alphas.append( alpha )
    #     q_deltas.append( q.run_stats[-1][ 'Error' ] )
    #     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
    #     q_means.append( q.run_stats[-1][ 'Mean V' ] )
    #     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )
    #     print( q.run_stats )

    # plt.plot( alphas, q_rewards )
    # plt.xlabel( 'Alpha' )
    # plt.ylabel( 'Mean Reward' )
    # plt.title( 'Frozen Lake: N = %s' % N )
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
    #     epsilons.append( epsilon )
    #     q_deltas.append( q.run_stats[-1][ 'Error' ] )
    #     q_iters.append( q.run_stats[-1][ 'Iteration' ] )
    #     q_means.append( q.run_stats[-1][ 'Mean V' ] )
    #     q_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in q.run_stats ] ) )
    #     print( q.run_stats )

    # plt.plot( epsilons, q_rewards )
    # plt.xlabel( 'Epsilon' )
    # plt.ylabel( 'Mean Reward' )
    # plt.title( 'Frozen Lake: N = %s' % N )
    # plt.legend( ['Q'] )
    # plt.show()

    from collections import defaultdict

    STATS = [ 'Error', 'Time', 'Max V', 'Mean V', 'Iteration' ]
    ALGOS = { 'PI': mdp.PolicyIteration,
            'VI': mdp.ValueIteration }
            # 'Q':  mdp.QLearning }

    results = defaultdict( defaultdict )
    for key, algo in ALGOS.items():
        fh = algo( P, R, 0.9 )
        if key == 'Q':
            fh = algo( P, R, 0.9, epsilon = 0.5 )
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
        plt.title( 'Frozen Lake: N = %s' % N )
        plt.legend( list( ALGOS.keys() ) )
        plt.show()

    plt.plot( results[ 'PI' ][ 'Time' ], results[ 'PI' ][ 'Error' ] )
    plt.plot( results[ 'VI' ][ 'Time' ], results[ 'VI' ][ 'Error' ] )
    # plt.plot( results[ 'Q' ][ 'Time' ], results[ 'Q' ][ 'Mean V' ] )
    plt.xlabel( 'Time' )
    plt.ylabel( 'Error' )
    plt.title( 'Frozen Lake: N = %s' % N )
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
    # PI_times = []
    # VI_times = []
    # import time
    # for gamma in [ 0.01 + 0.01 * i for i in range( 99 ) ]:
    #     pi = mdp.PolicyIteration( P, R, gamma )
    #     t0 = time.process_time()
    #     pi.run()
    #     print( 'PI Time: %s' % ( time.process_time() - t0 ) )
    #     gammas.append( gamma )
    #     PI_deltas.append( pi.run_stats[-1][ 'Error' ] )
    #     PI_iters.append( pi.run_stats[-1][ 'Iteration' ] )
    #     PI_means.append( pi.run_stats[-1][ 'Mean V' ] )
    #     PI_times.append( pi.run_stats[-1][ 'Time' ] )
    #     PI_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in pi.run_stats ] ) )
    #     print( pi.run_stats )

    #     vi = mdp.ValueIteration( P, R, gamma )
    #     t0 = time.process_time()
    #     vi.run()
    #     print( 'VI Time: %s' % ( time.process_time() - t0 ) )
    #     VI_deltas.append( vi.run_stats[-1][ 'Error' ] )
    #     VI_iters.append( vi.run_stats[-1][ 'Iteration' ] )
    #     VI_means.append( vi.run_stats[-1][ 'Mean V' ] )
    #     VI_times.append( vi.run_stats[-1][ 'Time' ] )
    #     VI_rewards.append( np.mean( [ run_stat[ 'Reward' ] for run_stat in vi.run_stats ] ) )


    # plt.plot( PI_iters, PI_times )
    # plt.plot( VI_iters, VI_times )
    # plt.xlabel( 'Iterations' )
    # plt.ylabel( 'Time' )
    # plt.title( 'Frozen Lake: N = %s' % N )
    # plt.legend( [ 'PI', 'VI' ] )
    # plt.show()

    # plt.plot( PI_times, PI_rewards )
    # plt.plot( VI_times, VI_rewards )
    # plt.xlabel( 'Time' )
    # plt.ylabel( 'Reward' )
    # plt.title( 'Frozen Lake: N = %s' % N )
    # plt.legend( [ 'PI', 'VI' ] )
    # plt.show()
    # # plt.plot( gammas, PI_rewards )
    # plt.plot( gammas, VI_rewards )
    # plt.xlabel( 'Gamma' )
    # plt.ylabel( 'Mean Reward' )
    # plt.title( 'Frozen Lake: N = %s' % N )
    # plt.legend( ['VI'] )
    # plt.show()

    # # plt.plot( gammas, PI_iters )
    # plt.plot( gammas, VI_iters )
    # plt.xlabel( 'Gamma' )
    # plt.ylabel( 'Iterations Needed for Convergence' )
    # plt.title( 'Frozen Lake: N = %s' % N )
    # plt.legend( ['VI'] )
    # plt.show()

if __name__ == '__main__':
    main()