#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/shuffle.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <random>
#include <cmath>
#include <iostream>
#include <thrust/sequence.h>

const unsigned int datapoints = 500, population_size = 1000;
const float a = -4, b = 2; // "a" is the left point "b" is the right
__constant__ float stddev = 1, mean = 0.5;
const float C0 = 2, C1 = 1, C2 = 1, C3 = -1, C4 = -0.5; // coefs 

typedef thrust::tuple<float, float, float, float, float> coefficients;


struct fitnesselement
{
	__device__
	float operator()(float& temp, coefficients& coefs)
	{
		float c0 = thrust::get<0>(coefs), c1 = thrust::get<1>(coefs), c2 = thrust::get<2>(coefs), c3 = thrust::get<3>(coefs), c4 = thrust::get<4>(coefs), x = temp;
		float fit = ((c4 * (x * x * x * x)) + (c3 * (x * x * x)) + (c2 * (x * x)) + (c1 * x) + c0);
		return	fit;
	}
};
template<typename T>
struct absolute_value
{
	__host__ __device__ T operator()(const T& x) const
	{
		return x < T(0) ? -x : x;
	}
};
struct crossover
{
	 __device__
	coefficients operator()(thrust::tuple<coefficients&, coefficients&>& temp, unsigned int seed)
	{
		 /*coefficients child = thrust::get<0>(temp);
		 thrust::get<2>(child) = thrust::get<2>(thrust::get<1>(temp));
		 thrust::get<3>(child) = thrust::get<3>(thrust::get<1>(temp));
		 thrust::get<4>(child) = thrust::get<4>(thrust::get<1>(temp));*/
		curandState state;
		curand_init(seed, 0, 0, &state);
		unsigned char n = static_cast<unsigned int>(curand_uniform(&state)*10000) % 33;
		coefficients child = thrust::get<0>(temp), parent2 = thrust::get<1>(temp);
		uint32_t mask = 0xffffffff, cast;
		mask >>= n;
		uint32_t* cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<0>(child));
		uint32_t* value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<0>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<4>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<4>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		return child;
	}
};
struct mutation
{
	__device__
	coefficients operator()(coefficients& temp, unsigned int seed)
	{
		curandState state;
		coefficients mutating_temp = temp;
		curand_init(seed, 0, 0, &state);
		uint32_t mask = 0, mask_ref = 0, cast = 0, value = 0;
		
		unsigned char k, n = static_cast<unsigned int>(curand_log_normal(&state, mean, stddev))%33;
		for (size_t i = 0; i < n; i++)
		{
			k = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
			mask_ref = 0x80000000;
			mask_ref >>= k;
			mask |= mask_ref;
		}
		uint32_t* cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<0>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? ((cast & ~0x7f800000)|(cast_ptr[0] & 0x7f800000)) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? ((cast & ~0x7f800000) | (cast_ptr[0] & 0x7f800000)) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? ((cast & ~0x7f800000) | (cast_ptr[0] & 0x7f800000)) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? ((cast & ~0x7f800000) | (cast_ptr[0] & 0x7f800000)) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<4>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? ((cast & ~0x7f800000) | (cast_ptr[0] & 0x7f800000)) : cast;
		return mutating_temp;
	}
};

int main()
{	
	thrust::device_vector<float> D_initial_dataset(datapoints);
	thrust::device_vector<float> step(datapoints);
	thrust::host_vector<coefficients> H_population(population_size, (0, 0, 0, 0, 0));
	float h = 0;
	h = (b-a) / static_cast<float>(datapoints);
	std::random_device rd;
	std::mt19937 gen(rd());
	for (size_t i = 0; i < datapoints; i++)
	{
		D_initial_dataset[i] = -(C4 * (a + i * h) * (a + i * h) * (a + i * h) * (a + i * h) + C3 * (a + i * h) * (a + i * h) * (a + i * h) + C2 * (a + i * h) * (a + i * h) + C1 * (a + i * h) + C0); //+ (static_cast<float>(gen()%2000)/1000 - 1); // y = C₄x⁴ + C₃x³ + C₂x² + C₁x¹ + C₀ + rnd(0,1)
		step[i] = (a + i * h);
	}
	
	for (size_t i = 0; i < population_size; i++)
	{
		thrust::get<0>(H_population[i]) = (static_cast<float>(gen() % 2) - 1) * static_cast<float>(gen() % 10) / 10;
		thrust::get<1>(H_population[i]) = (static_cast<float>(gen() % 2) - 1) * static_cast<float>(gen() % 10) / 10;
		thrust::get<2>(H_population[i]) = (static_cast<float>(gen() % 2) - 1)* static_cast<float>(gen() % 10) / 10;
		thrust::get<3>(H_population[i]) = (static_cast<float>(gen() % 2) - 1)* static_cast<float>(gen() % 10) / 10;
		thrust::get<4>(H_population[i]) = (static_cast<float>(gen() % 2) - 1)* static_cast<float>(gen() % 10) / 10;
	}
	//fitness vars

	thrust::device_vector<coefficients> D_population(population_size, (0,0,0,0,0));
	D_population = H_population;
	thrust::device_vector<float> deviation(datapoints);
	thrust::device_vector<float> fitness(population_size);
	thrust::device_vector<coefficients> best(1);

	//crossover vars

	thrust::device_vector<coefficients> shuffled_parents(population_size / 2);
	thrust::default_random_engine g;
	unsigned int generations = 1;
	std::cout << "Generations: ";
	std::cin >> generations;

	for (size_t j = 0; j < generations; j++)
	{
		// fitness calculation

		std::system("cls");
		std::cout << "Processing: " << j + 1 << '\n';
		H_population = D_population;
		std::cout << "Previous best: " << thrust::get<0>(H_population[0]) << ' ' << thrust::get<1>(H_population[0]) << ' ' << thrust::get<2>(H_population[0]) << ' ' << thrust::get<3>(H_population[0]) << ' ' << thrust::get<4>(H_population[0]) << "\nFitness: " << fitness[0] << '\n';
		for (size_t i = 0; i < population_size; i++)
		{
			thrust::transform(step.begin(), step.end(), thrust::make_constant_iterator(D_population[i]), deviation.begin(), fitnesselement());
			thrust::transform(D_initial_dataset.begin(), D_initial_dataset.end(), deviation.begin(), deviation.begin(), thrust::plus<float>());
			thrust::transform(deviation.begin(), deviation.end(),deviation.begin(), absolute_value<float>());
			fitness[i] = *thrust::max_element(deviation.begin(), deviation.end());
		}
		
		// selection

		thrust::sort_by_key(fitness.begin(), fitness.end(), D_population.begin());
		best[0] = D_population[0];

		//crossover generation

		thrust::shuffle_copy(thrust::device, D_population.begin(), D_population.begin() + population_size/2, shuffled_parents.begin(), g);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(D_population.begin(), shuffled_parents.begin())), thrust::make_zip_iterator(thrust::make_tuple(D_population.begin()+population_size/2, shuffled_parents.end())), thrust::counting_iterator<int>(0), D_population.begin()+population_size/2, crossover());
		thrust::shuffle_copy(thrust::device, D_population.begin(), D_population.begin() + population_size/2, shuffled_parents.begin(), g);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(D_population.begin(), shuffled_parents.begin())), thrust::make_zip_iterator(thrust::make_tuple(D_population.begin()+population_size/2, shuffled_parents.end())), thrust::counting_iterator<int>(0), D_population.begin(), crossover());
		
		//mutation

		thrust::transform(D_population.begin() + 1, D_population.end(), thrust::counting_iterator<int>(0), D_population.begin() + 1, mutation());
		thrust::copy_n(thrust::device, best.begin(), 1, D_population.begin());

		//thrust::sort_by_key(fitness.begin(), fitness.end(), D_population.begin());
		
	}
	

	std::cout << "\nBest fitted alternatives:\n";
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << thrust::get<0>(H_population[i]) << ' ';
		std::cout << thrust::get<1>(H_population[i]) << ' ';
		std::cout << thrust::get<2>(H_population[i]) << ' ';
		std::cout << thrust::get<3>(H_population[i]) << ' ';
		std::cout << thrust::get<4>(H_population[i]) << ' ';
		std::cout << "(" << fitness[i] << ")" << '\n';
	}
			
}
