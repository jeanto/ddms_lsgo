#ifndef DECOMPOSITION_LIBRARY_COMMON_H
#define DECOMPOSITION_LIBRARY_COMMON_H

#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <utility>

namespace decompose {
    typedef double scalar;

    enum class debug_level { None, VeryLow, Low, High };
    enum class migration_method { DDMS_TEDA, FIXED_BEST100, PROBA_BEST, FIXED_TEDA, PROBA_TEDA, DDMS_BEST, FIXED_BEST50 };

    enum class status {
        NotStarted = -1,
        Continue = 0,
        IterationLimit,
        EvaluationLimit,
        FOptimum
    };

    scalar max_limits();

    std::default_random_engine &default_generator();

    class options{
        protected:
            // Differential Grouping Method
            scalar dg_epsilon;
            // Fast Interdependency Identification Method
            scalar fii_epsilon_1;
            scalar fii_epsilon_2;
            scalar fii_sigma;
            scalar fii_delta;
            // Global Differential Grouping Method
            scalar gdg_alpha;
            size_t gdg_k;
            // Extended Differential Grouping Method
            scalar xdg_epsilon;
            // Random Grouping
            size_t rg_number_groups;
            // Recursive Differential Grouping 1 - Method
            scalar rdg_1_alpha;
            size_t rdg_1_k;
            // Recursive Differential Grouping 3 - Method
            size_t rdg_3_epsilon_n;
            // Differential Evolution - Cooperative Coevolutive Method
            scalar de_cr;
            size_t de_pop_size;
            size_t de_cycle;
        public:
            static options defaults();
            scalar get_dg_epsilon() const;
            scalar get_fii_epsilon_2() const;
            scalar get_fii_epsilon_1() const;
            scalar get_fii_sigma() const;
            scalar get_fii_delta() const;
            scalar get_gdg_alpha() const;
            scalar get_xdg_epsilon() const;
            size_t get_gdg_k() const;
            size_t get_rg_number_groups() const;
            size_t get_rdg_1_k() const;
            scalar get_rdg_1_alpha() const;
            size_t get_rdg_3_epsilon_n() const;
            scalar get_de_cr() const;
            size_t get_de_size_pop() const;
            size_t get_de_cycle() const;
            void set_dg_epsilon(scalar s);
            void set_fii_epsilon_1(scalar s);
            void set_fii_epsilon_2(scalar s);
            void set_fii_sigma(scalar s);
            void set_fii_delta(scalar s);
            void set_gdg_alpha(scalar s);
            void set_xdg_epsilon(scalar s);
            void set_gdg_k(size_t s);
            void set_rg_number_groups(size_t s);
            void set_rdg_1_k(size_t s);
            void set_rdg_1_alpha(scalar s);
            void set_rdg_3_epsilon_n(size_t s);
            void set_de_cr(scalar cr);
            void set_de_size_pop(size_t cr);
            void set_de_cycle(size_t cr);
    };

    class criteria{
        public:
            long iterations{};
            long evaluations{};
            bool fx_is_know{};
            scalar fx_best{};
            scalar error_fx_best{};
            scalar fun{};
            criteria();
            static criteria defaults();
            void reset();
            void print(std::ostream &os) const;
    };


    class stats{
        struct stat{
            scalar fx;
            size_t evaluation;
            std::chrono::duration<double> time;
        };

        protected:
            std::vector<stat> history;
            std::chrono::steady_clock::time_point starting_time;
        public:
            explicit stats() = default;
            void push(scalar fx, size_t e);
            std::vector<stat> get_history();
            void reset();
    };
}
#endif
