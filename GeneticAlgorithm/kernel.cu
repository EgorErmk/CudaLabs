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
__constant__ float a = -4, b = 2, mean = 2, stddev = 0.7; // "a" is the left point "b" is the right
const float C0 = 2, C1 = 1, C2 = 1, C3 = -1, C4 = -0.5; // coefs 

typedef thrust::tuple<float, float, float, float, float> coefficients;


struct fitnesselement
{
	__host__ __device__
	float operator()(thrust::tuple<float&, float&>& temp, coefficients& coefs)
	{
		return	thrust::get<0>(coefs) + thrust::get<1>(coefs) * thrust::get<1>(temp) + thrust::get<2>(coefs) * thrust::get<1>(temp) * thrust::get<1>(temp) + thrust::get<3>(coefs) * thrust::get<1>(temp) * thrust::get<1>(temp) * thrust::get<1>(temp) + thrust::get<4>(coefs) * thrust::get<1>(temp) * thrust::get<1>(temp) * thrust::get<1>(temp) * thrust::get<1>(temp) - thrust::get<0>(temp);
	}
};
template<typename T>
struct absolute_value
{
	__host__ __device__ 
	T 	operator()(const T& x) const
	{
		return x < T(0) ? -x : x;
	}
};
struct crossover
{
	 __device__
	coefficients operator()(thrust::tuple<coefficients&, coefficients&>& temp, unsigned int seed)
	{
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
		//skipahead(2, &state);
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		//skipahead(3, &state);
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		//skipahead(4, &state);
		n = static_cast<unsigned int>(curand_uniform(&state) * 10000) % 33;
		mask = 0xffffffff;
		mask >>= n;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(child));
		value_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(parent2));
		cast = cast_ptr[0];
		cast &= ~mask;
		cast |= value_ptr[0] & mask;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? cast_ptr[0] : cast;
		//skipahead(5, &state);
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
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? (cast & 0x7f800000) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<1>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? (cast & 0x7f800000) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<2>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? (cast & 0x7f800000) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<3>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? (cast & 0x7f800000) : cast;
		cast_ptr = reinterpret_cast<uint32_t*>(&thrust::get<4>(mutating_temp));
		cast = cast_ptr[0];
		value = ~cast & mask;
		cast &= ~mask;
		cast |= value;
		cast_ptr[0] = ((cast & 0x7f800000) == 0x7f800000) ? (cast & 0x7f800000) : cast;
		return mutating_temp;
	}
};

int main()
{
	thrust::host_vector<float> H_initial_dataset(datapoints, 0);
	thrust::device_vector<float> D_initial_dataset(datapoints);
	thrust::device_vector<float> step(datapoints);
	thrust::host_vector<coefficients> H_population(population_size, (0, 0, 0, 0, 0));
	float h = 0;
	h = (b-a) / static_cast<float>(datapoints);
	std::random_device rd;
	std::mt19937 gen(rd());
	for (size_t i = 0; i < datapoints; i++)
	{
		H_initial_dataset[i] = C4 * pow((a + i * h), 4) + C3 * pow((a + i * h), 3) + C2 * pow((a + i * h), 2) + C1 * (a + i * h) + C0; //+ (static_cast<float>(gen()%2000)/1000 - 1); // y = C₄x⁴ + C₃x³ + C₂x² + C₁x¹ + C₀ + rnd(0,1)
		step[i] = (a + i * h);
	}
	for (size_t i = 0; i < population_size; i++)
	{
		thrust::get<0>(H_population[i]) = static_cast<float>(gen() % 10) / 10;
		thrust::get<1>(H_population[i]) = static_cast<float>(gen() % 10) / 10;
		thrust::get<2>(H_population[i]) = static_cast<float>(gen() % 10) / 10;
		thrust::get<3>(H_population[i]) = static_cast<float>(gen() % 10) / 10;
		thrust::get<4>(H_population[i]) = static_cast<float>(gen() % 10) / 10;
	}
	D_initial_dataset = H_initial_dataset;

	//fitness vars

	thrust::device_vector<coefficients> D_population(population_size, (0,0,0,0,0));
	D_population = H_population;
	thrust::device_vector<float> deviation(datapoints);
	thrust::device_vector<float> fitness(population_size);
	thrust::device_vector<coefficients> parents(population_size / 2);

	//crossover vars

	thrust::device_vector<coefficients> shuffled_parents(population_size / 2);
	thrust::device_vector<coefficients> children(population_size / 2);
	thrust::default_random_engine g;
	unsigned int generations = 1;
	std::cout << "Generations: ";
	std::cin >> generations;

	for (size_t j = 0; j < generations; j++)
	{
		// fitness calculation
		std::system("cls");
		std::cout << "Processing: " << j + 1 << '\n';
		H_population[0] = D_population[0];
		std::cout << "Previous best: " << thrust::get<0>(H_population[0]) << ' ' << thrust::get<1>(H_population[0]) << ' ' << thrust::get<2>(H_population[0]) << ' ' << thrust::get<3>(H_population[0]) << ' ' << thrust::get<4>(H_population[0]) << '\n';
		for (size_t i = 0; i < population_size; i++)
		{
			thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(D_initial_dataset.begin(), step.begin())), thrust::make_zip_iterator(thrust::make_tuple(D_initial_dataset.end(), step.end())), thrust::make_constant_iterator(D_population[i]), deviation.begin(), fitnesselement());
			fitness[i] = thrust::transform_reduce(deviation.begin(), deviation.end(), absolute_value<float>(), 0, thrust::maximum<float>());
		}

		// selection
		thrust::stable_sort(thrust::device,fitness.begin(), fitness.end());
		thrust::sort_by_key(fitness.begin(), fitness.end(), D_population.begin());
		thrust::copy_n(thrust::device, D_population.begin(), population_size / 2, parents.begin());

		//crossover generation

		thrust::shuffle_copy(thrust::device, parents.begin(), parents.end(), shuffled_parents.begin(), g);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(parents.begin(), shuffled_parents.begin())), thrust::make_zip_iterator(thrust::make_tuple(parents.end(), shuffled_parents.end())), thrust::counting_iterator<int>(0), children.begin(), crossover());
		thrust::copy_n(thrust::device, children.begin(), population_size / 2, D_population.begin());
		thrust::shuffle_copy(thrust::device, parents.begin(), parents.end(), shuffled_parents.begin(), g);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(parents.begin(), shuffled_parents.begin())), thrust::make_zip_iterator(thrust::make_tuple(parents.end(), shuffled_parents.end())), thrust::counting_iterator<int>(0), children.begin(), crossover());
		thrust::copy_n(thrust::device, children.begin(), population_size / 2, D_population.begin() + population_size / 2);
		//mutation
		thrust::transform(D_population.begin() + 1, D_population.end(), thrust::counting_iterator<int>(0), D_population.begin() + 1, mutation());
		thrust::copy_n(thrust::device, parents.begin(), 1, D_population.begin());

		//thrust::sort_by_key(fitness.begin(), fitness.end(), D_population.begin());
		
	}
	H_population = D_population;
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << thrust::get<0>(H_population[i]) << ' ';
		std::cout << thrust::get<1>(H_population[i]) << ' ';
		std::cout << thrust::get<2>(H_population[i]) << ' ';
		std::cout << thrust::get<3>(H_population[i]) << ' ';
		std::cout << thrust::get<4>(H_population[i]) << '\n';
	}
			
}
