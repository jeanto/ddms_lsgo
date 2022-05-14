#include "problems/f1_cec.h"
#include "problems/f2_cec.h"
#include "problems/f3_cec.h"
#include "problems/f4_cec.h"
#include "problems/f5_cec.h"
#include "problems/f6_cec.h"
#include "problems/f7_cec.h"
#include "problems/f8_cec.h"
#include "problems/f9_cec.h"
#include "problems/f10_cec.h"
#include "problems/f11_cec.h"
#include "problems/f12_cec.h"
#include "problems/f13_cec.h"
#include "problems/f14_cec.h"
#include "problems/f15_cec.h"
#include "differential_evolution_cooperative_coevolutive.h"
#include "distributed_differential_evolution_cooperative_coevolutive.h"
#include <iostream>
#include <string>
#include <climits>
#include <iomanip>

using namespace decompose;
using namespace std;

class read_files{
    public:
        string data_dir = "decomposition_solutions";

        void read_decomposition_solutions(const size_t id_function, const string &method, optimization_problem &problem, criteria &current_criteria) const{
            string name_file = "F" + to_string(id_function) + "_" + method + ".txt";

            stringstream ss;
            ss << data_dir <<"/" << method << "/" << name_file;
            ifstream file(ss.str());

            cout << "NAME FILE: " << ss.str() << endl;

            std::string tmp;
            // Function Description
            file >> tmp;
            cout << tmp << " ";
            file >> tmp;
            cout << tmp << endl;

            // Method Description
            file >> tmp;
            cout << tmp << " ";
            file >> tmp;
            cout << tmp << endl;

            // Function Evaluation Description
            file >> tmp;
            cout << tmp << " ";
            long fes;
            file >> fes;
            cout << fes << endl;
            current_criteria.evaluations = fes;

            // Subproblems Description
            file >> tmp;
            file >> tmp;

            string sub_problem_tmp;
            bool add;
            std::vector<std::set<size_t>> sub_problems;
            size_t id_sub_problem = -1;
            for(char & iter : tmp){
                if((char) iter == '['){
                    id_sub_problem++;
                    std::set<size_t> set_tmp;
                    add = false;
                    sub_problem_tmp = "";
                    sub_problems.push_back(set_tmp);
                }
                else if ((char) iter == ']' || (char) iter == ',') {
                    add = true;
                }
                else if ((char) iter != ' '){
                    add = false;
                    sub_problem_tmp += iter;
                }
                if(add){
                    scalar value = stoi(sub_problem_tmp);
                    sub_problems[id_sub_problem].insert(value);
                    sub_problem_tmp = "";
                }
            }

            // Setting subproblems
            problem.set_problem_structure(sub_problems);
            file.close();

            // Checking Subproblems
            string structure_method;
            structure_method += "[";
            for(size_t i = 0; i < sub_problems.size(); i++){
                structure_method += "[";
                auto it = sub_problems[i].begin();
                bool print = true;
                while(print && !sub_problems.empty()){
                    structure_method += (std::to_string(*it));
                    it++;
                    if(it != sub_problems[i].end()){
                        structure_method += ",";
                    }else{
                        print = false;
                    }
                }
                structure_method += "]";
            }
            structure_method += "]";
            cout << "SUBPROBLEMS SIZE: " << sub_problems.size() << endl;
            if(structure_method.find(tmp) != std::string::npos){
                cout << "CHECKING SUBPROBLEMS: OK" << endl;
            }
            else{
                cerr << "CHECKING SUBPROBLEMS: ERROR" << endl;
                exit(0);
            }
        }
};

void save_results(const string &id_version, size_t id_func, size_t idx, vector<scalar> &x, const scalar fx, int island, double time){
    ofstream file_results;
    ofstream file_solutions;
    file_results.open(id_version + ".txt", std::ofstream::app);
    file_results << id_version << "; " << idx << "; " << island << "; " << id_func << "; " << time << "; " << setprecision(20) << fx << endl;
    file_results.close();
    // << id_rep << "; 3e6; "
    // file_solutions.open(id_version + "_solutions.txt", std::ofstream::app);
    // file_solutions << id_version << "; " << "Func: " << id_func << "; Rep: " << id_rep << "; ";
    // for(scalar i : x){
    //     file_solutions << i << "; ";
    // }
    // file_solutions << endl;
    // file_solutions.close();
}

int main(int argc, char** argv) {
    const size_t max_id_function = atoi(argv[1]);   // number of bench function
    const size_t algo = atoi(argv[2]); // 0: DDMS_TEDA, 1: FIXED_BEST100, 2: PROBA_BEST, 3: FIXED_TEDA, 4: PROBA_TEDA, 5: DDMS_BEST, 6: FIXED_BEST50.
    const size_t idx = atoi(argv[3]);

    const size_t max_rep = 1;   // 30;
    string methods[] = {"DG"};  //string methods[] = {"DG", "DG2", "XDG", "FII", "GDG", "RDG", "RDG2", "RDG3"};

    //for(size_t id_function = 1; id_function <= max_id_function; id_function++){
    for(size_t id_function = max_id_function; id_function <= max_id_function; id_function++){
        for(const auto & method : methods){
            for(size_t i_rep = 1; i_rep <= max_rep; i_rep++) {
                
                //string results_file = "results_" + method + std::to_string(algo);
                string method_file = method + std::to_string(algo);
                size_t dimension;
                scalar lower_bound, upper_bound;
                if (id_function == 13 || id_function == 14) {
                    dimension = 905;
                } else {
                    dimension = 1000;
                }
                criteria current_, stop_;
                options options_ = options::defaults();
                stop_.evaluations = 3e6;// 2e5; // stop_.evaluations = 3e6;
                stop_.iterations = LONG_MAX;
                stop_.fx_is_know = true;
                stop_.error_fx_best = 0.0;
                //stop_.error_fx_best = 1e-8;
                stop_.fx_best = 0.0;

                switch (id_function) { 
                    case 1: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f1_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);
                        //solver.set_debug(debug_level::VeryNone);
                        //solver.set_debug(debug_level::None);
                        solver.set_migration_method(migration_method::FIXED_BEST);
                        solver.minimize(f, x0);
                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, 0);
                        break;
                    }
                    case 2: {
                        lower_bound = -5.0;
                        upper_bound = 5.0;
                        f2_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);
                        solver.minimize(f, x0);
                        x0 = solver.get_best_solution();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, 0, 0);
                        break;
                    }
                    case 3: {
                        lower_bound = -32.0;
                        upper_bound = 32.0;
                        f3_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);
                        solver.minimize(f, x0);
                        x0 = solver.get_best_solution();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, 0, 0);
                        break;
                    }
                    case 4: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f4_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);
                        
                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 5: {
                        lower_bound = -5.0;
                        upper_bound = 5.0;
                        f5_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 6: {
                        lower_bound = -32.0;
                        upper_bound = 32.0;
                        f6_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);                        
                        
                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 7: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f7_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 8: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f8_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);
                        
                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 9: {
                        lower_bound = -5.0;
                        upper_bound = 5.0;
                        f9_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 10: {
                        lower_bound = -32.0;
                        upper_bound = 32.0;
                        f10_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 11: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f11_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 12: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f12_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 13: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f13_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 14: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f14_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    case 15: {
                        lower_bound = -100.0;
                        upper_bound = 100.0;
                        f15_cec f(dimension, vector<scalar>(dimension, lower_bound),vector<scalar>(dimension, upper_bound));

                        read_files read_files_;
                        read_files_.read_decomposition_solutions(id_function, method, f, current_);
                        cout << endl << endl;

                        vector<scalar> x0(dimension, 0.0);
                        //differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        distributed_differential_evolution_cooperative_coevolutive solver(current_, stop_, options_);
                        solver.set_debug(debug_level::None);

                        // set method
                        migration_method method_cast = static_cast<migration_method>(algo);
                        solver.set_migration_method(method_cast);

                        const clock_t begin_time = clock();
                        solver.minimize(f, x0);
                        const clock_t end_time  = clock();
                        scalar timei 		    = double(end_time - begin_time) / CLOCKS_PER_SEC;

                        x0 = solver.get_best_solution();
                        int island = solver.get_rank();
                        unsigned long i_last = solver.get_stats().get_history().size() - 1;
                        scalar fx_best = solver.get_stats().get_history()[i_last].fx;
                        save_results(method_file, id_function, idx, x0, fx_best, island, timei);
                        break;
                    }
                    default: {
                        cerr << "Invalid id function." << endl;
                        exit(2);
                    }
                }
            }
        }
    }
    return 0;
}