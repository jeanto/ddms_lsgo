#ifndef DECOMPOSITION_LIBRARY_DISTRIBUTED_SHADE_COOPERATIVE_COEVOLUTIVE_H
#define DECOMPOSITION_LIBRARY_DISTRIBUTED_SHADE_COOPERATIVE_COEVOLUTIVE_H

#include <decomposition/solver.h>
#include <decomposition/optimization_problem.h>
#include <vector>

using namespace decompose;

class distributed_shade_cooperative_coevolutive : public solver {
    public:
        using super = solver;

        distributed_shade_cooperative_coevolutive(criteria &current_criteria, criteria &stop_criteria, options &o);

    protected:
        std::vector<std::vector<scalar>> pop;
        std::vector<std::vector<scalar>> pop_aux;
        std::vector<scalar> best_solution;
        scalar fx_best_solution{};
        size_t id_best_solution{};
        std::vector<scalar> fx_pop;
        std::vector<scalar> fx_pop_aux;
        scalar de_cr{};
        size_t de_size_pop{};
        size_t de_cycle{};
        size_t dimension{};

        int POOL = 0;   // POOL id
        int rank;       // process id
        int size;       // number of process

    public:
        distributed_shade_cooperative_coevolutive() = default;
        void minimize(optimization_problem &problem_, std::vector<scalar> &x0) override;
        std::vector<scalar> get_best_solution() const;
        void setting_method(size_t d);
        void update_best_solution(std::vector<scalar> &x, scalar fx, size_t i);
        void allocate_initial_population();
        void generate_evaluate_init_population(optimization_problem &problem);
        static void generate_random_index(std::vector<size_t> &v, size_t n, size_t min, size_t max);
        static scalar get_bounds(scalar x, scalar min_bound, scalar max_bound);
        void differential_mutation_operator(optimization_problem &problem, size_t i_ind, size_t i_x, std::vector<size_t> &index);
        void evolution(optimization_problem &problem, size_t index_sub_problem);
        void ddms_evolution(optimization_problem &problem, size_t index_sub_problem);
        void fixed_proba_evolution(optimization_problem &problem, size_t index_sub_problem);
        int get_rank() const;
        std::string get_method(migration_method m);

    // node struct
    struct node {
        size_t node_id;
        std::vector<scalar> x;
        scalar fitness;
        scalar distance;
    };

    struct node_arc {
        size_t NP;					// number of individuals
        size_t arc_ind_count;		// number of filled positions
        std::vector<node> pop;	    // archived individuals
    };

    // convergence struct 
    struct conv {
        std::vector<scalar> m;	 		// mean vector
        std::vector<scalar> std;	 	// std vector
        std::vector<scalar> MR;	 		// mean vector, it is the value of m just before the last diversity
                                            // enhancement operation
        std::vector<size_t> gamma;	 	// gamma is set to 1 to indicate that the population has converged 
                                            // in the jth dimention
        std::vector<scalar> theta; 		// |m - MR|*T if std <= T; T otherwise
        std::vector<scalar> omega; 		// if std is not greater than omega, gamma is set to 1. min(T, theta)
    };

    // stagnation struct
    struct stag {
        std::vector<scalar> m_previous;    	// mean vector (G - 1)
        std::vector<scalar> std_previous;  	// std vector (G - 1)
        std::vector<size_t> lambda;        	// number of successive generations where the values of mjg 
                                                // and stdjg remain unchanged
        std::vector<size_t> gamma;   	    // denote whether the population has stagnated in the jth 
                                                // dimension at the gth generation
    };

    // enhancement struct
    struct enhan {
        std::vector<size_t> gamma;	// flag to denote whether the population needs to be rediversified in the jth dimension at the gth generation
        size_t zg;				    // flag to denote whether the population is rediversified at the gth generation
        size_t mig;				    // flat ot denote whether it is necessary to make migration
    };

    // tedacloud struct
    struct tedacloud {
        size_t id;
        scalar sk;          		// samples
        scalar vk;          		// variance;
        std::vector<scalar> uk;  	// center (mean)
        std::vector<node> xk;  		// samples belonging to the group 
        node best_teda;
    };


    // AEPD: Auto-Enhanced Population Diversity (AEPD)
    protected:
        conv conv_stats;			// convergence stats
        stag stag_stats;			// stagnation stats
        enhan enhan_stats;			// enhancement stats
        scalar T	= 1e-3; 		// min(T,thetajg), where T = 10^-3
        scalar c	= 1e-3; 		// the population will be also rediversified with a small 
                                    // 		probability in any dimension j where rjg = 1
        size_t UN{};                // UN = SUBPOP_SIZE, the larger population size, the more 
			                        //		generations it will take for the population to 
			                        // 		enter a stable stagnation rate.

        // TEDA params and variables
        std::vector<tedacloud> clouds;      // TEDA Clouds 
        std::vector<node> cloud_inds;       // return TEDA individuals on teda_cloud funcion  
        size_t window = 10;                 // maximun number of individuals to keep inside the cloud
        size_t max_n_clouds = 10;           // maximun number of clouds
        size_t k_teda = 0;                  // stream TEDA
        node ind_received;                  // individual received in the migration
        const scalar r0 = 0.001;

        // FIXED and PROBA params and variables
        size_t FIXED_INTERVAL100    = 100;    // as Apolloni (2008, 2014), it migrates every 100 generations
        size_t FIXED_INTERVAL50     = 50;     // as Meng (2017), it migrates every 50 generations
        scalar PROBA_INTERVAL       = 0.05;   // probabilist migration

        const size_t memory_size    = 5;    
        const scalar arc_rate       = 1.4;
        const scalar PI             = 3.1415926535897932384626433832795029;
        const scalar p_best_rate 	= 0.11;
        int arc_size;	
        int memory_pos              = 0;

        std::vector<node> popr;     // population of each process
        std::vector<node> children; // children of each process
        node_arc archive;		    // archive struct
        std::vector<scalar> memory_sf {0.5,0.5,0.5,0.5,0.5};
        std::vector<scalar> memory_cr {0.5,0.5,0.5,0.5,0.5};

    public:
        void convergence(std::set<size_t> &sub_problem);
        void stagnation(std::set<size_t> &sub_problem, size_t isub);
        void need_rediversify(std::set<size_t> &sub_problem);
        scalar rand_0_1();
        void teda_cloud(node ind_received, optimization_problem &problem);
        void update_cloud(size_t c, node ind);
        scalar distance(std::vector<scalar> x, std::vector<scalar> uk);
        std::vector<scalar> ls_process(node pop_ls, std::vector<scalar> best_point, optimization_problem &problem);

        // SHADE
        void shade(optimization_problem &problem, std::set<size_t> &sub_problem);
        scalar gauss(scalar mu, scalar sigma);
        scalar cauchy_g(scalar mu, scalar gamma);
        static bool compare(node a, node b);
        void operateCurrentToPBest1BinWithArchive(int &target, int &p_best_individual, scalar &scaling_factor, scalar &cross_rate,
                    std::set<size_t> &sub_problem, std::vector<scalar> lower_bound, std::vector<scalar> upper_bound);

};

#endif // DECOMPOSITION_LIBRARY_DISTRIBUTED_SHADE_COOPERATIVE_COEVOLUTIVE_H
