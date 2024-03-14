#include "pool_approx.h"
#include "maxpool_convex_hull.h"

size_t handle_pool_layer(elina_manager_t *man, elina_abstract0_t *element,
						 size_t *pool_size, size_t *input_size, size_t *strides, size_t pad_top, size_t pad_left, size_t pad_bottom, size_t pad_right, size_t *output_size, size_t *predecessors, size_t num_predecessors, bool is_maxpool)
{
	assert(num_predecessors == 1);
	assert(pool_size[2] == 1);
	// assert(stride[0]==2 && stride[1]==2 && stride[2]==1);

	size_t i, j, k;

	size_t num_input_neurons = input_size[0] * input_size[1] * input_size[2];
	size_t num_out_neurons = output_size[0] * output_size[1] * output_size[2];

	size_t o12 = output_size[1] * output_size[2];
	size_t i12 = input_size[1] * input_size[2];
	size_t p01 = pool_size[0] * pool_size[1];

	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp, num_out_neurons, predecessors, num_predecessors, false);
	size_t out_pos;
	double *inf = (double *)calloc(p01, sizeof(double));
	double *sup = (double *)calloc(p01, sizeof(double));
	size_t *pool_map = (size_t *)calloc(p01, sizeof(size_t));
	neuron_t **out_neurons = fp->layers[numlayers]->neurons;
	int k1 = predecessors[0] - 1;
	neuron_t **in_neurons = fp->layers[k1]->neurons;
	for (out_pos = 0; out_pos < num_out_neurons; out_pos++)
	{
		size_t out_x = out_pos / o12;
		size_t out_y = (out_pos - out_x * o12) / output_size[2];
		size_t out_z = out_pos - out_x * o12 - out_y * output_size[2];
		// size_t inp_x = out_x*pool_size[0];
		// size_t inp_y = out_y*pool_size[1];
		size_t inp_z = out_z;
		// size_t inp_pos = inp_x*i12 + inp_y*input_size[2] + inp_z;
		// size_t pool_start_dim = out_pos*pool_size[0]*pool_size[1];
		// printf("inpXYZ: %zu, %zu, %zu %zu %zu\n", inp_x, inp_y, inp_z, out_pos, num_out_neurons);
		// printf("outXYZ: %zu, %zu, %zu\n", out_x, out_y, out_z);
		// fflush(stdout);
		size_t x_shift, y_shift, l = 0;
		double max_u = -INFINITY;
		double max_l = -INFINITY;
		double max_u2 = -INFINITY;
		size_t max_u_var2 = 0.0;
		double max_u1l = -INFINITY;
		double max_m = -INFINITY;
		size_t max_m_var = 0.0;
		size_t max_l_var = 0.0;
		size_t max_u_var = 0.0;
		for (x_shift = 0; x_shift < pool_size[0]; x_shift++)
		{
			for (y_shift = 0; y_shift < pool_size[1]; y_shift++)
			{
				long int x_val = out_x * strides[0] + x_shift - pad_top;
				if (x_val < 0 || x_val >= (long int)input_size[0])
				{
					continue;
				}
				long int y_val = out_y * strides[1] + y_shift - pad_left;
				if (y_val < 0 || y_val >= (long int)input_size[1])
				{
					continue;
				}
				size_t pool_cur_dim = x_val * i12 + y_val * input_size[2] + inp_z;
				if (pool_cur_dim >= num_input_neurons)
				{
					continue;
				}
				pool_map[l] = pool_cur_dim;
				// use the ReLU bounds from the previous layer
				double lb = -in_neurons[pool_cur_dim]->lb;
				double ub = in_neurons[pool_cur_dim]->ub;
				inf[l] = lb;
				sup[l] = ub;

				if (sup[l] > max_u)
				{
					max_u = sup[l];
					max_u_var = pool_map[l];
					max_u1l = inf[l];
				}
				if (inf[l] > max_l)
				{
					max_l = inf[l];
					max_l_var = pool_map[l];
				}
				if ((inf[l] + sup[l]) / 2 > max_m)
				{
					max_m = (inf[l] + sup[l]) / 2;
					max_m_var = pool_map[l]; /*select the maximum m*/
				}
				l++;
			}
		}

		for (i = 0; i < l; i++)
		{ /*select the second maximum u*/
			if (pool_map[i] == max_u_var)
			{
				continue;
			}
			if (sup[i] > max_u2)
			{
				// max_temp = ;
				max_u_var2 = pool_map[i];
				max_u2 = sup[i];
			}
		}
		double ucoeff[1];
		size_t udim[1];
		double lcoeff[1];
		size_t ldim[1];
		// printf("max_l: %gmax_u: %g\n",max_l,max_u);
		// fflush(stdout);
		if (is_maxpool)
		{
			// bool flag = false;
			//     	size_t var = 0;

			if (max_l >= max_u2)&&(max_u1l>=max_l)
			{
				// if(0){
				// x_new = x_var
				// printf("out_pos: %zu\n",out_pos);
				// fflush(stdout);

				ucoeff[0] = 1;
				udim[0] = max_u_var;
				out_neurons[out_pos]->lexpr = create_sparse_expr(ucoeff, 0, udim, 1);
				out_neurons[out_pos]->uexpr = create_sparse_expr(ucoeff, 0, udim, 1);

				// out_neurons[out_pos]->expr = create_sparse_expr(coeff,0,dim,1);
			}
			else
			{
				ucoeff[0] = (double)(max_u - max_u2) / (max_u - max_u1l);
				udim[0] = max_u_var;
				ldim[0] = max_m_var;
				lcoeff[0] = 1;
				out_neurons[out_pos]->uexpr = create_sparse_expr(ucoeff, max_u2 + (max_u - max_u2) / (max_u - max_u1l) * (-max_u1l), udim, 1);
				out_neurons[out_pos]->lexpr = create_sparse_expr(lcoeff, 0.0, ldim, 1);
				// out_neurons[out_pos]->uexpr = create_sparse_expr(ucoeff,max_l-sum_l,udim,p01);
				// sort_sparse_expr(out_neurons[out_pos]->uexpr);
				// free(ucoeff);
				// free(udim);
				// out_neurons[out_pos]->lexpr = create_cst_expr(-max_l,max_l);
				// out_neurons[out_pos]->uexpr = create_cst_expr(-max_u,max_u);
			}
			out_neurons[out_pos]->lb = -max_l;
			out_neurons[out_pos]->ub = max_u;
		}
		else
		{
			size_t j;
			double *lcoeff = (double *)malloc(l * sizeof(double));
			double *ucoeff = (double *)malloc(l * sizeof(double));
			size_t *udim = (size_t *)malloc(l * sizeof(size_t));
			double avg_l = 0.0, avg_u = 0.0;
			fesetround(FE_DOWNWARD);
			double coeff_u = 1.0 / (double)(l);
			fesetround(FE_UPWARD);
			double coeff_l = 1.0 / (double)(l);
			for (j = 0; j < l; j++)
			{
				ucoeff[j] = coeff_u;
				lcoeff[j] = coeff_l;
				udim[j] = pool_map[j];
				avg_l = avg_l + inf[j] * coeff_l;
				avg_u = avg_u + sup[j] * coeff_u;
			}
			double cst = 0.0;
			out_neurons[out_pos]->lexpr = create_sparse_expr(lcoeff, cst, udim, l);
			out_neurons[out_pos]->uexpr = create_sparse_expr(ucoeff, cst, udim, l);
			out_neurons[out_pos]->lb = -avg_l;
			out_neurons[out_pos]->ub = avg_u;
			free(ucoeff);
			free(lcoeff);
			free(udim);
		}
	}

	// update_state_using_previous_layers_parallel(man,fp,numlayers);
	free(inf);
	free(sup);
	free(pool_map);
	// free(output_size);
	// fflush(stdout);
	// fppoly_fprint(stdout,man,fp,NULL);
	// fflush(stdout);
	return num_out_neurons;
}
