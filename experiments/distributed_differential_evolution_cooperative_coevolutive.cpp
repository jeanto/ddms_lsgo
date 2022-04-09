#include "distributed_differential_evolution_cooperative_coevolutive.h"
#include <vector>
#include <random>
#include <iomanip>
#include <mpi.h>

distributed_differential_evolution_cooperative_coevolutive::distributed_differential_evolution_cooperative_coevolutive(criteria &current_criteria, criteria &stop_criteria, options &o) : super(current_criteria, stop_criteria, o){}

void distributed_differential_evolution_cooperative_coevolutive::minimize(optimization_problem &problem, std::vector<scalar> &x0) {
    setting_method(problem.get_dimension());
    allocate_initial_population();
    generate_evaluate_init_population(problem);
    const long max_evaluation = this->stop_criteria.evaluations;
    bool is_known_problem_structure = problem.is_known_problem_structure();
    if(m_debug >= debug_level::VeryLow){
        std::cout << "Solver Initialization ..." << std::endl;
        std::cout << "Iteration: " << current_criteria.iterations
                  << " - Evaluations: " << current_criteria.evaluations
                  << " - Fx Best: " << std::setprecision(20) << fx_best_solution << std::endl;
    }
    if(!is_known_problem_structure){
        std::cerr << "Problem Structure is not Available." << std::endl;
        return;
    }
    size_t index_sub_problem = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);

	int is_to_exit_program = 0;
	MPI_Request exitProgramRequest = MPI_REQUEST_NULL;

    MPI_Irecv(&is_to_exit_program, 1, MPI_INT, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, &exitProgramRequest);

    status status_ = check_convergence(stop_criteria, current_criteria);
    while(status_ == status::Continue){

        size_t max_eval_tmp = (this->current_criteria.evaluations + de_cycle <= max_evaluation ? this->current_criteria.evaluations + de_cycle : max_evaluation);
        this->stop_criteria.evaluations = max_eval_tmp;
        scalar gain = fx_best_solution;

        // Se subproblema eh igual a 1, chama evolution()
        //evolution(problem, index_sub_problem);
        if(this->m_migration_method == migration_method::FIXED_BEST || this->m_migration_method == migration_method::PROBA_BEST){
            fixed_proba_evolution(problem, index_sub_problem);
        } else {
            ddms_evolution(problem, index_sub_problem);
        }

        //MPI_Barrier(MPI_COMM_WORLD);
        
        gain = gain - fx_best_solution;
        this->stop_criteria.evaluations = max_evaluation;
        index_sub_problem = (index_sub_problem+1) % problem.get_problem_structure().size();
        if(m_debug >= debug_level::None){
            scalar perc_exec = double(current_criteria.evaluations) / double(stop_criteria.evaluations);
            std::cout << get_method(this->m_migration_method) << "%: " << std::setprecision(5)
                    << perc_exec << "%"
                    << " #Eval: " << current_criteria.evaluations << " - " << stop_criteria.evaluations
                    << " #Fx: " << fx_best_solution
                    << " #Gain: " << gain 
                    << " #Rank: " << rank << std::endl;            
        }
        status_ = check_convergence(stop_criteria, current_criteria);

        if (status_ != status::Continue){
            is_to_exit_program = 1;
            for (int i = 0; i < size; i++){
                if (rank != i){
                    MPI_Send(&is_to_exit_program, 1, MPI_INT, i, 5, MPI_COMM_WORLD);			
                }			
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    
        // Check if some island has finished
        MPI_Test(&exitProgramRequest, &is_to_exit_program, MPI_STATUS_IGNORE);
        if (is_to_exit_program != 0) {
            if(this->m_debug >= debug_level::VeryLow) {
                std::cout << "[" << rank << "] alguem terminou o programa. Saindo do programa!!!" << std::endl;
            }
            break;
        }
    }

    std::cout << std::endl << std::endl << "[" << rank << "] Solver Status: " << std::endl << get_status_string(status_) << std::endl;
    std::cout << "Fx Best Solution: " << fx_best_solution << std::endl << std::endl;
    // Tratar aqui para encerrar!

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

void distributed_differential_evolution_cooperative_coevolutive::setting_method(size_t d){
    de_cr = this->solver_options.get_de_cr();
    de_size_pop = this->solver_options.get_de_size_pop();
    de_cycle = this->solver_options.get_de_cycle();
    dimension = d;
    best_solution = std::vector<scalar>(dimension, 1.0);
    fx_best_solution = max_limits();
}

void distributed_differential_evolution_cooperative_coevolutive::update_best_solution(std::vector<scalar> &x, scalar fx, size_t i){
    if(fx < fx_best_solution){
        this->m_status = check_convergence(this->stop_criteria, this->current_criteria);
        if(this->m_status == status::Continue){
            best_solution = x;
            fx_best_solution = fx;
            id_best_solution = i;
            this->m_stats.push(fx, this->current_criteria.evaluations);
        }
    }
}

void distributed_differential_evolution_cooperative_coevolutive::allocate_initial_population(){
    pop.reserve(de_size_pop);
    pop_aux.reserve(de_size_pop);
    for(size_t i = 0; i < de_size_pop; i++){
        std::vector<scalar> x(dimension, 0.0);
        pop.push_back(x);
        pop_aux.push_back(x);
    }
    fx_pop = std::vector<scalar>(de_size_pop, 1.0);
    fx_pop_aux = std::vector<scalar>(de_size_pop, 1.0);
}

void distributed_differential_evolution_cooperative_coevolutive::generate_evaluate_init_population(optimization_problem &problem){
    for(size_t i = 0; i < de_size_pop; i++){
        for (size_t j = 0; j < dimension; j++) {
            std::uniform_real_distribution<scalar> dist(problem.get_lower_bound()[j], problem.get_upper_bound()[j]);
                pop[i][j] = dist(default_generator());
                pop_aux[i][j] = pop[i][j];
        }
        scalar fx = problem.value(pop[i]);
        ++this->current_criteria.evaluations;
        fx_pop[i] = fx;
        fx_pop_aux[i] = fx;
        update_best_solution(pop[i], fx, i);
    }
}

void distributed_differential_evolution_cooperative_coevolutive::generate_random_index(std::vector<size_t> &v, size_t n, size_t min, size_t max){
    std::uniform_int_distribution<int> dist_pop(min, max);
    for(size_t i = 1; i < n; i++){
        bool diff_index;
        do{
            diff_index = true;
            v[i] = dist_pop(default_generator());
            if(v[0] == v[i]){
                diff_index = false;
            }
            else{
                for(size_t j = 1; j < i; j++){
                    if(v[j] == v[i]) {
                        diff_index = false;
                        j += i;
                    }
                }
            }
        }while(!diff_index);
    }
}

scalar distributed_differential_evolution_cooperative_coevolutive::get_bounds(scalar x, scalar min_bound, scalar max_bound){
    if(x > max_bound){
        return max_bound;
    }else if(x < min_bound){
        return min_bound;
    } else {
        return x;
    }
}

// 0: DDMS_TEDA, 1: FIXED_BEST, 2: PROBA_BEST, 3: FIXED_TEDA, 4: PROBA_TEDA, 5: DDMS_BEST
std::string distributed_differential_evolution_cooperative_coevolutive::get_method(migration_method m){
    std::string met;
    if(m == migration_method::DDMS_TEDA){
        met = "[ DDMS_TEDA ] ";
    }   
    else if(m == migration_method::FIXED_BEST){
        met = "[ FIXED_BEST ] ";
    }
    else if(m == migration_method::PROBA_BEST){
        met = "[ PROBA_BEST ] ";
    }
    else if(m == migration_method::FIXED_TEDA){
        met = "[ FIXED_TEDA ] ";
    }
    else if(m == migration_method::PROBA_TEDA){
        met = "[ PROBA_TEDA ] ";
    }
    else if(m == migration_method::DDMS_BEST){
        met = "[ DDMS_BEST ] ";
    }
    return met;
}

void distributed_differential_evolution_cooperative_coevolutive::differential_mutation_operator(
    optimization_problem &problem, size_t i_ind, size_t i_x, std::vector<size_t> &index){
    std::uniform_real_distribution<scalar> dist(0.5, 1.0);
    scalar de_f = dist(default_generator());
    pop_aux[i_ind][i_x] = pop[index[1]][i_x] + de_f * (pop[index[2]][i_x] - pop[index[3]][i_x]);
    pop_aux[i_ind][i_x] = get_bounds(pop_aux[i_ind][i_x], problem.get_lower_bound()[i_x], problem.get_upper_bound()[i_x]);
}

/*
  DDMS
  Version: 3.0   Date: 13/Jan/2022
  Written by Jean Nunes (jean.to[at]gmail.com)

    MPI.TAGs:
        0: some island reached the stop criteria
        1: island request a migration (send its best individuo)
        2: pool sending some individuo for some island
        3: pool sending signal to islands break (exit)

    mpirun --use-hwthread-cpus -np 4 experiments_de_cc

    Fix:
        - algoritmo trava no fim
        - adicionar o aepd e a migracao (com o melhor e o random)
*/
void distributed_differential_evolution_cooperative_coevolutive::ddms_evolution(optimization_problem &problem, size_t index_sub_problem){
    std::set<size_t> sub_problem = problem.get_problem_structure()[index_sub_problem];
    std::uniform_int_distribution<int> dist_dim(0, dimension-1);
    std::uniform_real_distribution<scalar> dist_cr(0.0, 1.0);
    const size_t n_solutions = 4;
    std::vector<size_t> index(n_solutions);

    size_t isub = 0; // keep number of iterations on subproblem
    if(this->m_debug >= debug_level::Low) {
        std::cout << "Current Iteration: " << this->current_criteria.iterations
                  << " - Evaluations: " << this->current_criteria.evaluations
                  << " - Fx: " << fx_best_solution << std::endl;
    }

    // island keeps a broadcasting conection openned to receive signal to exit
	int exit_item   = 0;
    int is_to_exit  = 0;
    int flag_send   = 0;	// flag to check send of request
    int flag_recv   = 0;	// flat to check receive of request
	MPI_Request exitRequest = MPI_REQUEST_NULL;
    MPI_Request myRequestSend[2];
    MPI_Request myRequest;

    //MPI_Barrier(MPI_COMM_WORLD);

    // islands keeps a broadcasting openned to receive signal to exit
	if (rank != POOL){
		MPI_Irecv(&exit_item, 1, MPI_INT, POOL, 4, MPI_COMM_WORLD, &exitRequest);
	}

    this->m_status = check_convergence(this->stop_criteria, this->current_criteria);
    while(this->m_status == status::Continue){
        
        isub++;                     // increment number of iterations

        int rank_source;            // rank      from the target island
        std::vector<double> ind; 	// individuo from the target island
        ind.resize(dimension);

        if (rank == POOL){
			long nfe_island;
			MPI_Status myStatus;
			MPI_Irecv(&rank_source, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &myRequest);
            if(this->m_debug >= debug_level::VeryLow) {
			    std::cout << "[" << rank << "] esperando chegar..." << std::endl;
            }
			MPI_Wait(&myRequest, &myStatus);

            // send broadcasting to stop because some island reached some stop criterion
            if (myStatus.MPI_TAG != 0){
                int exit_signal = myStatus.MPI_TAG;
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout 	<< "[" << rank << "]" << " diz que [" << rank_source 
				    		<< "] terminou. Vou avisar a todos... " << std::endl;
                }
				for (int i = 1; i < size; i++){
					MPI_Send(&exit_signal, 1, MPI_INT, i, 4, MPI_COMM_WORLD);						
				}
                break;
            }
            else{
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout << "[" << rank << "]" << " migracao solicitada por [" << rank_source << "]" << std::endl;
                }

                MPI_Recv(ind.data(), dimension, MPI_DOUBLE, rank_source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                //MPI_Irecv(&rank_source, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &myRequest);
                //MPI_Recv(&nfe_island, 1, MPI_LONG, rank_source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 0: DDMS_TEDA, 3: FIXED_TEDA, 4: PROBA_TEDA
                if (this->m_migration_method != migration_method::DDMS_BEST){
                    //std::cout << "[teda1]" << std::endl;
                    // TEDA Cloud
                    ind_received.x = ind;
                    teda_cloud(ind_received, problem);
                    //std::cout << "[teda2]" << std::endl;

                    //size_t id_send = rand() % cloud_inds.size();
                    MPI_Isend(cloud_inds[0].x.data(), dimension, MPI_DOUBLE, rank_source, 0, MPI_COMM_WORLD, &myRequest);

                    int send_ok = 0;
                    do{
                        MPI_Test(&myRequest, &send_ok, MPI_STATUS_IGNORE);
                        if (!send_ok){
                            MPI_Cancel(&myRequest);
                            MPI_Request_free(&myRequest);
                            break;
                        }
                    }while (send_ok != 1);
                }
                // DDMS_BEST
                else {
                    scalar fx = problem.value(ind);
                    ++this->current_criteria.evaluations;
                    update_best_solution(ind, fx, rank);
                    //MPI_Send(best_solution.data(), dimension, MPI_DOUBLE, rank_source, 0, MPI_COMM_WORLD);

                    MPI_Isend(best_solution.data(), dimension, MPI_DOUBLE, rank_source, 0, MPI_COMM_WORLD, &myRequest);
                    int send_ok = 0;
                    do{
                        MPI_Test(&myRequest, &send_ok, MPI_STATUS_IGNORE);
                        if (!send_ok){
                            MPI_Cancel(&myRequest);
                            MPI_Request_free(&myRequest);
                            break;
                        }
                    }while (send_ok != 1);
                }

                if(this->m_debug >= debug_level::VeryLow) {
                    std::cout 	<< "[" << rank << "] enviou um individuo para : [" << rank_source 
                            << "]" << "[" << current_criteria.evaluations << "]" << std::endl;
                }
            }
        }
        else{
            
            // Check if some island has finished
			MPI_Test(&exitRequest, &is_to_exit, MPI_STATUS_IGNORE);
			if (is_to_exit != 0) {
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout << "[" << rank << "] alguem terminou. Saindo..." << std::endl;
                }
				break;
			}

            for(size_t i = 0; i < de_size_pop; i++){
                index[0] = i;
                generate_random_index(index, n_solutions, 0, de_size_pop-1);
                size_t r = dist_dim(default_generator());
                pop_aux[i] = pop[i];
                for(unsigned long j : sub_problem){
                    if(j == r || dist_cr(default_generator()) <= de_cr){
                        differential_mutation_operator(problem, i, j, index);
                    }
                }
                scalar fx = problem.value(pop_aux[i]);
                ++this->current_criteria.evaluations;
                fx_pop_aux[i] = fx;
                update_best_solution(pop_aux[i], fx, i);
            }
            for(size_t i = 0; i < de_size_pop; i++){
                if(fx_pop_aux[i] < fx_pop[i]){
                    pop[i] = pop_aux[i];
                    fx_pop[i] = fx_pop_aux[i];
                }
            }

            ++this->current_criteria.iterations;
            this->current_criteria.fx_best = fx_best_solution;
            this->m_status = check_convergence(this->stop_criteria, this->current_criteria);            
            if(this->m_debug >= debug_level::VeryLow) {
                std::cout << "Current Iteration: " << this->current_criteria.iterations
                        << " - Evaluations: " << this->current_criteria.evaluations
                        << " - Fx: " << fx_best_solution 
                        << " - Rank: " << rank << std::endl;
            }

            // Check if some island has finished
			MPI_Test(&exitRequest, &is_to_exit, MPI_STATUS_IGNORE);
			if (is_to_exit != 0) {
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout << "[" << rank << "] alguem terminou. Saindo..." << std::endl;
                }
				break;
			}            

            // a stop criterian has been reached, the process has terminated
            if (this->m_status != status::Continue){
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout << "[" << rank << "] terminei!" << int(this->m_status) << std::endl;                
                }
                int tag_to_exit = int(this->m_status);

				MPI_Isend(&rank, 1, MPI_INT, POOL, tag_to_exit, MPI_COMM_WORLD, &myRequest);	
                MPI_Test(&myRequest, &flag_send, MPI_STATUS_IGNORE);
                //MPI_Send(&rank, 1, MPI_INT, POOL, tag_to_exit, MPI_COMM_WORLD);
                MPI_Wait(&exitRequest, MPI_STATUS_IGNORE);        

                break;
            }

            // FIXED_TEDA
            if(this->m_migration_method == migration_method::FIXED_TEDA) {
                if (current_criteria.iterations % (FIXED_INTERVAL+rank) == 0) {
                    enhan_stats.zg = 1;
                } else {
                    enhan_stats.zg = 0;
                }
            } 
            // FIXED_BEST
            else if (this->m_migration_method == migration_method::PROBA_TEDA) {
                if (rand_0_1() < PROBA_INTERVAL){
                    enhan_stats.zg = 1;
                } else {
                    enhan_stats.zg = 0;
                }
            }
            // DDMS_TEDA ou DDMS_BEST
            else {
                convergence(sub_problem);       // check convergence
                stagnation(sub_problem, isub);        // check stagnation
                need_rediversify(sub_problem);  // check rules
                enhan_stats.zg = 0;
            }

            /* DO MIGRATION:  || rand_0_1() < 0.05 */
            if (enhan_stats.zg == 1){
                
                if(this->m_debug >= debug_level::VeryLow) {
				    std::cout << "[" << rank << "] vou pedir para [" << POOL << "]" << std::endl;
                }

				int tag_send = 0;
				MPI_Isend(&rank, 1, MPI_INT, POOL, tag_send, MPI_COMM_WORLD, &myRequestSend[0]);
				MPI_Isend(best_solution.data(), dimension, MPI_DOUBLE, POOL, tag_send, MPI_COMM_WORLD, &myRequestSend[1]);
				//MPI_Isend(&current_criteria.evaluations, 1, MPI_LONG, POOL, tag_send, MPI_COMM_WORLD, &myRequestSend[2]);

				int exit_signal = 0;
                flag_send = 0;
				do{
					is_to_exit = 0;
					MPI_Test(&exitRequest, &is_to_exit, MPI_STATUS_IGNORE);
					if (is_to_exit != 0){
						exit_signal = -10;
						MPI_Cancel(myRequestSend);
						MPI_Request_free(myRequestSend);
						break;
					}

					MPI_Testall(2, myRequestSend, &flag_send, MPI_STATUS_IGNORE);
				}while (flag_send != 1);

                // some island terminated
				if (exit_signal < 0) break;

                if(this->m_debug >= debug_level::VeryLow) {
                    std::cout 	<< "[" << rank << "] acabou de pedir para [" << POOL << "]" << std::endl;
                }

                std::vector<scalar> ind_mig;
				ind_mig.reserve(dimension); 
                MPI_Request myRequestNewInd;

                MPI_Irecv(ind_mig.data(), dimension, MPI_DOUBLE, POOL, 0, MPI_COMM_WORLD, &myRequestNewInd);

				flag_recv = 0;
				do{
					is_to_exit = 0;
					MPI_Test(&exitRequest, &is_to_exit, MPI_STATUS_IGNORE);
					if (is_to_exit != 0) {
						exit_signal = -10;
						MPI_Cancel(&myRequestNewInd);
						MPI_Request_free(&myRequestNewInd);	
						break;
					}

					MPI_Test(&myRequestNewInd, &flag_recv, MPI_STATUS_IGNORE);
				} while (flag_recv != 1);

                // some island terminated
				if (exit_signal < 0) break;

                if(this->m_debug >= debug_level::VeryLow) {
                    std::cout << "[" << rank << "] recebeu de [" << POOL << "]" << std::endl;
                }

                scalar fx_mig = problem.value(ind_mig);
                ++this->current_criteria.evaluations;
                
                // avoid current best individual is pruned
                size_t id = rand() % de_size_pop;
                while (id_best_solution == id) {
                    id = rand() % de_size_pop;
                }
                pop[id] = ind_mig;
                fx_pop[id] = fx_mig;

                update_best_solution(ind_mig, fx_mig, id);               
            }
        }
    }

    if(this->m_debug >= debug_level::VeryLow) {
	    std::cout << "[" << rank << "] saindo..." << std::endl;
    }
    //MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<scalar> distributed_differential_evolution_cooperative_coevolutive::get_best_solution() const{
    return this->best_solution;
}

int distributed_differential_evolution_cooperative_coevolutive::get_rank() const{
    return this->rank;
}

scalar distributed_differential_evolution_cooperative_coevolutive::rand_0_1() {
	scalar r = static_cast <scalar> (rand()) / static_cast <scalar> (RAND_MAX);
	return r;
}

void distributed_differential_evolution_cooperative_coevolutive::convergence(std::set<size_t> &sub_problem){
	
    std::vector<scalar> mjg(sub_problem.size(), 0.0); 	   // media das variaveis dos individuos na jth dimensao na gth geracao 
	std::vector<scalar> stdjg(sub_problem.size(), 0.0);    // desvio padrao das variaveis dos individuos na jth dimensao na gth geracao
	std::vector<size_t> gamajg(sub_problem.size(), 0);	   // gamma is set to 1 to indicate that the population has converged in the jth dimention
	std::vector<scalar> thetajg(sub_problem.size(), 0.0);  // |m - MR|*T if std <= T; T otherwise
	std::vector<scalar> omegajg(sub_problem.size(), 1.0);  // if std is not greater than omega, gamma is set to 1. min(T, theta)

    size_t j = 0;
    size_t sk = 1;
    // mean
    for (size_t i = 0; i < de_size_pop; i++){
        j = 0;
        for(unsigned long k : sub_problem){
            mjg[j] = (((sk - 1.0) / sk) * mjg[j]) + (pop[i][k] / sk);
            j++;
        }
        sk++;
    }

    // standard deviation
    for (size_t i = 0; i < de_size_pop; i++){
        j = 0;
        for(unsigned long k : sub_problem){
            stdjg[j] = stdjg[j] + std::pow(pop[i][k] - mjg[j],2);
            j++;
        }
    }

    j = 0;
	for (unsigned long k : sub_problem){
		stdjg[j] = std::sqrt(stdjg[j] / de_size_pop);
        j++;
	}

	// get previous mean and std
	if (current_criteria.iterations > 1){
		stag_stats.m_previous   = conv_stats.m;
		stag_stats.std_previous = conv_stats.std;		
	}
   
	conv_stats.m 	 = mjg;
	conv_stats.std 	 = stdjg;
	conv_stats.gamma = gamajg;
	conv_stats.theta = thetajg;
	conv_stats.omega = omegajg;

	// the initial value of MRj is set to the mean value mjg of the initialized population
	if (current_criteria.iterations == 1){
		conv_stats.MR = conv_stats.m;
	}

    j = 0;
	for (unsigned long k : sub_problem){
		// Theta				
		if (conv_stats.std[j] <= T)
			conv_stats.theta[j] = (std::fabs(conv_stats.m[j] - (conv_stats.MR[j])) * T);
		else
			conv_stats.theta[j] = T;

		// Omega
		conv_stats.omega[j] = std::min(T, conv_stats.theta[j]);

		// Gamma
		if (conv_stats.std[j] <= conv_stats.omega[j])
			conv_stats.gamma[j] = 1;
		else
			conv_stats.gamma[j] = 0;

        j++;
	}
}

void distributed_differential_evolution_cooperative_coevolutive::stagnation(std::set<size_t> &sub_problem, size_t isub){

    //std::cout << current_criteria.iterations << std::endl;
	if (isub == 1){
		std::vector<size_t> gamajg(sub_problem.size(), 0);
		std::vector<size_t> lambdajg(sub_problem.size(), 0);
		stag_stats.gamma  = gamajg;
		stag_stats.lambda = lambdajg;
	}
	else{
        size_t j = 0;
		for (unsigned long k : sub_problem){
			if((conv_stats.m[j] == stag_stats.m_previous[j]) && (conv_stats.std[j] == stag_stats.std_previous[j])){
				stag_stats.lambda[j] = stag_stats.lambda[j] + 1;
			}
			else{
				stag_stats.lambda[j] = 0;
			}
            j++;
		}
	}
    
    UN = de_size_pop;
    size_t j = 0;
	for (unsigned long k : sub_problem){
		if (stag_stats.lambda[j] >= UN){
			stag_stats.gamma[j] = 1;
		}
		else{
			stag_stats.gamma[j] = 0;
		}
        j++;
	}
}
    
void distributed_differential_evolution_cooperative_coevolutive::need_rediversify(std::set<size_t> &sub_problem){

	std::vector<size_t> gamma(sub_problem.size(), 0);
	enhan_stats.gamma = gamma;

	// rediversify with a small probability
	enhan_stats.zg = 0;

	size_t sum_rjg = 0;
    size_t j = 0;
	for (unsigned long k : sub_problem){
		if (conv_stats.gamma[j] == 1 || stag_stats.gamma[j] == 1){
			sum_rjg++;
			enhan_stats.gamma[j] = 1;
		}
        j++;
	}

    // probabilidade migracao vai aumentando conforme a evolução
	scalar conv_stag = sum_rjg/(scalar)sub_problem.size();
	scalar prob_mig  = (1.0-(current_criteria.evaluations/(scalar)stop_criteria.evaluations));
	if (conv_stag >= prob_mig){
		enhan_stats.zg = 1;
	}
    else{
        if (rand_0_1() < c){
		    enhan_stats.zg = 1;
        }
    }
    //std::cout << "------{" << enhan_stats.zg << "} " << conv_stag << " " << prob_mig << " " << sub_problem.size() << "-----" << std::endl;
}


// TEDA CLOUD
void distributed_differential_evolution_cooperative_coevolutive::teda_cloud(node ind, optimization_problem &problem){

    std::vector<node> teda_migra; // vector with inds to be migrated

    bool outlier_in_all_clouds = false;
    scalar norm_eccentricity = 0;
    k_teda++;

    if (k_teda == 1){
        tedacloud cloud0;

		// create cloud 0 and add x1
		cloud0.id = 1;
		cloud0.vk = 0.0;
		cloud0.uk = ind.x;
		cloud0.sk = 1;
		cloud0.xk.push_back(ind);
		clouds.push_back(cloud0);

		cloud_inds.push_back(ind);
    }
    else{
        outlier_in_all_clouds = true;
        size_t n_cloud = clouds.size();
        size_t new_ind = 0;
        for(int c = n_cloud - 1; c >= 0; c--){
            tedacloud c_cloud = clouds[c];

            scalar sk = c_cloud.sk + 1;

            // update mean of the cloud
			std::vector<scalar> uk;
			uk.resize(dimension);
			for (size_t j = 0; j < dimension; j++){
				uk[j] = (((sk - 1.0) / sk) * c_cloud.uk[j]) + (ind.x[j] / sk);
			}

			// update variance of the cloud
			scalar dist = distance(ind.x, uk);
			scalar vk 	= (((sk - 1.0) / sk) * c_cloud.vk) + ((1.0 / (sk - 1.0)) * dist);

            // eccentricity
            scalar ecc 	= (1 / sk) + (dist / (sk * vk));

            // normalized eccentricity 
            norm_eccentricity 	= ecc / 2.0;

            // typicality 
            scalar tip 	= 1.0 - ecc;

            // check if it is outlier 
            scalar m_param = 1.0;
			scalar thr_out;
			if (sk == 2.0){
				thr_out = ((std::pow(m_param, 2.0) + 1.0) / 4.0);
			}
			else{
				thr_out = ((std::pow(m_param, 2.0) + 1.0) / (2.0 * sk));
			}

            bool is_outlier = false;
			if (norm_eccentricity > thr_out) {
				if (sk == 2.0){
					if (c_cloud.vk < r0){
						is_outlier = true;
					}    
				}  
				else is_outlier = true;
			}      

            // it is not outlier, update cloud c 
            if (!is_outlier){
                c_cloud.sk++;
				c_cloud.uk = uk;
				c_cloud.vk = vk;
                // add
				if (c_cloud.xk.size() < window)
					c_cloud.xk.push_back(ind);
                // slide window
				else{
					std::vector<node> xks(c_cloud.xk.begin() + 1, c_cloud.xk.end());
					xks.push_back(ind);
					c_cloud.xk = xks;
				}
				clouds[c] = c_cloud;

				// find its place, not create new cloud
				outlier_in_all_clouds = false;                
            }
            else{
                // send just one individuo
                if (new_ind == 0){
                    node migra;
                    migra.x = clouds[c].uk;
                    cloud_inds.push_back(migra);
                    new_ind++;
                }
            }
        }
         
        //std::cout << "teda7" << std::endl;

        // it is outlier in all clouds
        if (outlier_in_all_clouds){
            if (n_cloud < max_n_clouds){
				tedacloud cloudi;
				cloudi.id = n_cloud+1;
				// create cloud i and add x1
				cloudi.vk = 0.0;
				cloudi.uk = ind.x;
				cloudi.sk = 1.0;
				cloudi.xk.push_back(ind);

				clouds.push_back(cloudi);                
            }
        }
        // put individual in a random cloud, good to keep diversity
        else{
            size_t idc = rand() % n_cloud;
            if (clouds[idc].xk.size() < window)
                clouds[idc].xk.push_back(ind);
            else{
                std::vector<node> xks(clouds[idc].xk.begin() + 1, clouds[idc].xk.end());
                xks.push_back(ind);
                clouds[idc].xk = xks;
            }
            update_cloud(idc, ind);
        }
        

		// individuo is not outlier in any cloud
		if (new_ind == 0){
            for(size_t c = 0; c < clouds.size(); c++){
                for(size_t i = 0; i < clouds[c].xk.size(); i++){
                    clouds[c].xk[i].x = ls_process(clouds[c].xk[i], best_solution, problem);
                }         
            }
            // send mean ind from a random cloud
            size_t c = rand() % n_cloud;
            node migra;
            migra.x = clouds[c].uk;
            cloud_inds.push_back(migra);
        }
    }
}

void distributed_differential_evolution_cooperative_coevolutive::update_cloud(size_t c, node ind){
    scalar sk = clouds[c].sk + 1;

    // update mean of cloud c
    std::vector<scalar> uk;
    uk.resize(dimension);
    for (size_t j = 0; j < dimension; j++){
        uk[j] = (((sk - 1.0) / sk) * clouds[c].uk[j]) + (ind.x[j] / sk);
    }

    // update variance of cloud c
    scalar dist = distance(ind.x, uk);
    scalar vk 	= (((sk - 1.0) / sk) * clouds[c].vk) + ((1.0 / (sk - 1.0)) * dist);    
    clouds[c].sk++;
    clouds[c].uk = uk;
    clouds[c].vk = vk;
}

scalar distributed_differential_evolution_cooperative_coevolutive::distance(std::vector<scalar> x, std::vector<scalar> uk){
    scalar sum = 0.0;
    scalar distance = 0.0;

    for (size_t j = 0; j < dimension; j++){
        sum = sum + std::pow((x[j] - uk[j]),2.0);
    }
    distance = std::sqrt(sum);  
    return distance;  
}

std::vector<scalar> distributed_differential_evolution_cooperative_coevolutive::ls_process(node pop_ls, std::vector<scalar> best_point, optimization_problem &problem){

    size_t igen = 1;
    std::vector<scalar> xp;
    xp.resize(dimension);
    for (size_t i = 0; i < dimension; i++){
        std::default_random_engine gen1;
        std::normal_distribution<scalar> distribution(best_point[i],(std::log(igen)/igen)*(std::fabs(pop_ls.x[i]-best_point[i])));        
        scalar xi = distribution(gen1);

        std::normal_distribution<scalar> randn1(0,1); 

        std::default_random_engine gen2;
        std::default_random_engine gen3;
        scalar add = randn1(gen2)*best_point[i] - randn1(gen3)*pop_ls.x[i];

        scalar point = xi + add;

        if (point > problem.get_upper_bound()[i] || point < problem.get_lower_bound()[i]){
            point = (problem.get_upper_bound()[i] - problem.get_lower_bound()[i]) * rand_0_1() + problem.get_lower_bound()[i];
        }

        xp[i] = point;
            
    }

    return xp;
}

void distributed_differential_evolution_cooperative_coevolutive::fixed_proba_evolution(optimization_problem &problem, size_t index_sub_problem){
    std::set<size_t> sub_problem = problem.get_problem_structure()[index_sub_problem];
    std::uniform_int_distribution<int> dist_dim(0, dimension-1);
    std::uniform_real_distribution<scalar> dist_cr(0.0, 1.0);
    const size_t n_solutions = 4;
    std::vector<size_t> index(n_solutions);
    //std::cout   << "------- FIXED_PROBA EVOLUTION -------" << std::endl;
    if(this->m_debug >= debug_level::Low) {
        std::cout << "Current Iteration: " << this->current_criteria.iterations
                  << " - Evaluations: " << this->current_criteria.evaluations
                  << " - Fx: " << fx_best_solution << std::endl;
    }

    srand(time(NULL) + rank);

	// compute the next and previous nodes
	int next 		= (rank + 1) % size;
	int prev 		= (size + rank - 1) % size;

    // island keeps a broadcasting conection openned to receive signal to exit
	int is_to_exit_subproblem;
    int flag_exit   = 0;	// flag to exit from evolution loop
    int exit        = 0;
	MPI_Request exitRequest = MPI_REQUEST_NULL;
    MPI_Request myRequest;

    MPI_Barrier(MPI_COMM_WORLD);

    // islands keeps a broadcasting openned to receive signal to exit
	//if (rank != POOL){
	//MPI_Irecv(&exit_item, 1, MPI_INT, POOL, 3, MPI_COMM_WORLD, &exitRequest);
    MPI_Irecv(&is_to_exit_subproblem, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &exitRequest);
	//}

    this->m_status = check_convergence(this->stop_criteria, this->current_criteria);
    while(this->m_status == status::Continue){

        for(size_t i = 0; i < de_size_pop; i++){
            index[0] = i;
            generate_random_index(index, n_solutions, 0, de_size_pop-1);
            size_t r = dist_dim(default_generator());
            pop_aux[i] = pop[i];
            for(unsigned long j : sub_problem){
                if(j == r || dist_cr(default_generator()) <= de_cr){
                    differential_mutation_operator(problem, i, j, index);
                }
            }
            scalar fx = problem.value(pop_aux[i]);
            ++this->current_criteria.evaluations;
            fx_pop_aux[i] = fx;
            update_best_solution(pop_aux[i], fx, i);
        }
        for(size_t i = 0; i < de_size_pop; i++){
            if(fx_pop_aux[i] < fx_pop[i]){
                pop[i] = pop_aux[i];
                fx_pop[i] = fx_pop_aux[i];
            }
        }

        ++this->current_criteria.iterations;
        this->current_criteria.fx_best = fx_best_solution;
        this->m_status = check_convergence(this->stop_criteria, this->current_criteria);            
        if(this->m_debug >= debug_level::VeryLow) {
            std::cout << "Current Iteration: " << this->current_criteria.iterations
                    << " - Evaluations: " << this->current_criteria.evaluations
                    << " - Fx: " << fx_best_solution 
                    << " - Rank: " << rank << std::endl;
        }  

        // send to the neighbor island (forward) a migration code
        // 1 -> it will be made a migration; 0 -> otherwise
        int migration_code_recv = 0; 
        if(this->m_migration_method == migration_method::FIXED_BEST) {
            if (current_criteria.iterations % (FIXED_INTERVAL+rank) == 0) {
                migration_code_recv = 1;
            } else {
                migration_code_recv = 0;
            }
        }
        else if (this->m_migration_method == migration_method::PROBA_BEST) {
            if (rand_0_1() < PROBA_INTERVAL){
                migration_code_recv = 1;
            } else {
                migration_code_recv = 0;
            }
        }

		MPI_Send(&migration_code_recv, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        //std::cout << "[" << rank << "] enviou beacon {" << migration_code_recv << "}" << " para [" << next << "] " << std::endl;
        int migration_code_send = 0;
        MPI_Recv(&migration_code_send, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //std::cout << "[" << rank << "] recebeu beacon {" << migration_code_send << "}" << " de [" << prev << "] " << std::endl;

        /* do migration -> send the best individual to the neighbor island (behind) */
        if (migration_code_send == 1){
                
            if(this->m_debug >= debug_level::VeryLow) {
                std::cout << "[" << rank << "] vai enviar para [" << prev << "] um individuo" << std::endl;
            }

            MPI_Send(best_solution.data(), dimension, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD);

            //std::cout << "[" << rank << "] enviou para [" << prev << "] um individuo" << std::endl;
        }

        /* receive migration -> receive the best individual from the neighbor island (forward) */
        if (migration_code_recv == 1) {

            std::vector<scalar> ind_mig;
            ind_mig.reserve(dimension); 

            MPI_Recv(ind_mig.data(), dimension, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(this->m_debug >= debug_level::VeryLow) {
                std::cout << "[" << rank << "] recebeu de [" << next << "]" << std::endl;
            }

            scalar fx_mig = problem.value(ind_mig);
            ++this->current_criteria.evaluations;
            
            // avoid current best individual is pruned
            size_t id = rand() % de_size_pop;
            while (id_best_solution == id) {
                id = rand() % de_size_pop;
            }
            pop[id] = ind_mig;
            fx_pop[id] = fx_mig;

            update_best_solution(ind_mig, fx_mig, id);    
        }

        if (this->m_status != status::Continue){
            if(this->m_debug >= debug_level::VeryLow) {
                // a stop criterian has been reached, the process has terminated
                std::cout << "[" << rank << "] terminei!" << std::endl;                
            }
            is_to_exit_subproblem = 1;
            for (int i = 0; i < size; i++){
                if (rank != i){
                    MPI_Send(&is_to_exit_subproblem, 1, MPI_INT, i, 4, MPI_COMM_WORLD);			
                }			
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    
        // Check if some island has finished
        MPI_Test(&exitRequest, &is_to_exit_subproblem, MPI_STATUS_IGNORE);
        if (is_to_exit_subproblem != 0) {
            if(this->m_debug >= debug_level::VeryLow) {
                std::cout << "[" << rank << "] alguem terminou o subproblema. Saindo do subproblema!!!" << std::endl;
            }
            break;
        }

        // exit signal to exit of the loop
        // MPI_Bcast(&flag_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // if (flag_exit == 1){
        //     // a stop criterian has been reached, the process has terminated
        //     std::cout << "[" << rank << "] saindo!" << std::endl;
        //     break;
        // }
    }

    if(this->m_debug >= debug_level::VeryLow) {
	    std::cout << "[" << rank << "] saindo..." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}