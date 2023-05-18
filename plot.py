import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

def plot_init_samples(penalized):
	init_eval_lhs = np.genfromtxt('../../outputs/lhs/all_true_eval.dat')[3:103]
	init_eval_lhs_feas_idx = np.where(init_eval_lhs[:,-1]==0.0)[0]
	init_eval_dcgan = np.genfromtxt('../../outputs/dcgan/all_true_eval.dat')[3:103]
	init_eval_dcgan_feas_idx = np.where(init_eval_lhs[:,-1]==0.0)[0]
	base_indiv = np.genfromtxt('../../outputs/lhs/all_true_eval.dat')[0]
	plt.axvspan(-10,110,0.0,1.0,facecolor='darkgrey', alpha=0.5)
	
	if penalized:
		pass
	else:
		base = base_indiv[0]*10000
		for i in range(2):
			if i == 0:
				init_eval = init_eval_lhs
				init_eval_feas_idx = init_eval_lhs_feas_idx
				feas_sym = 'ro'
				infeas_sym = 'rx'
				feas_legend = 'LHS - Feasible'
				infeas_legend = 'LHS - Infeasible'
			else:
				init_eval = init_eval_dcgan
				init_eval_feas_idx = init_eval_dcgan_feas_idx
				feas_sym = 'bo'
				infeas_sym = 'bx'
				feas_legend = 'DCGAN - Feasible'
				infeas_legend = 'DCGAN - Infeasible'
			eval_counter = 1
			feas_label = True
			infeas_label = True
			feas = 0
			for indiv in range(len(init_eval)):
				if indiv in init_eval_feas_idx:
					feas+=1
					if feas_label:
						plt.plot(eval_counter,init_eval[indiv,0]*10000,feas_sym,markersize=3,label=feas_legend)
						feas_label = False
					else:
						plt.plot(eval_counter,init_eval[indiv,0]*10000,feas_sym,markersize=3)
				else:
					if infeas_label:
						plt.plot(eval_counter,init_eval[indiv,0]*10000,infeas_sym,markersize=6,label=infeas_legend)
						infeas_label = False
					else:
						plt.plot(eval_counter,init_eval[indiv,0]*10000,infeas_sym,markersize=6)
				eval_counter += 1
			print(feas)

	plt.plot(1,base,'o',color='slategrey',label='Baseline')
	plt.hlines(base,0,100,color='slategrey',linestyle='dashed')

	plt.xlabel('Number of CFD evaluations')
	if penalized:
		plt.ylabel('Penalized objective (count)')
	else:
		plt.ylabel('Drag coefficient (count)')
	plt.title('Initial samples')
	plt.legend(loc='upper right')
	plt.xlim([-10, 110])
	plt.ylim([50, 700])
	plt.grid(True,color='white')
	# plt.show()
	# plt.savefig('initial_samples.png',format='png',dpi=300)

def plot_performance(method):
	all_true_eval = np.genfromtxt(f'../../outputs/{method}/all_true_eval.dat')[3:]
	obj_min = np.genfromtxt(f'../../outputs/{method}/obj_min.dat')
	all_true_eval_feas_idx = np.where(all_true_eval[:,-1]==0.0)[0]
	base_indiv = np.genfromtxt(f'../../outputs/{method}/all_true_eval.dat')[0]
	plt.axvspan(-10,100,0.0,1.0,facecolor='darkgrey', alpha=0.5)
	plt.axvspan(100,1000,0.0,1.0,facecolor='lightgrey', alpha=0.5)

	eval_counter = 1
	feas_label = True
	infeas_label = True
	for indiv in range(len(all_true_eval)):
		penalized_obj = all_true_eval[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval[indiv,2])))*10000
		if indiv in all_true_eval_feas_idx:
			if feas_label:
				plt.plot(eval_counter,penalized_obj,'bo',markersize=3,label='Feasible Sol')
				feas_label = False
			else:
				plt.plot(eval_counter,penalized_obj,'bo',markersize=3)
		else:
			if infeas_label:
				plt.plot(eval_counter,penalized_obj,'ro',markersize=3,label='Infeasible Sol')
				infeas_label = False
			else:
				plt.plot(eval_counter,penalized_obj,'ro',markersize=3)
		eval_counter += 1

	penalized_base = base_indiv[0]*10000 + np.abs(np.max((0.0,base_indiv[1])))*1000 + np.abs(np.max((0.0,base_indiv[1])))*10000
	plt.plot(obj_min[:,0],obj_min[:,1],'k-',label='Feas min obj')
	plt.plot(1,penalized_base,'o',color='slategrey',label='Baseline')
	plt.hlines(penalized_base,0,1000,color='slategrey',linestyle='dashed')
	plt.xlabel('Number of CFD evaluations')
	plt.ylabel('Penalized drag coefficient (count)')
	plt.title('Penalized objective function vs No of true eval')
	plt.legend(loc='upper right')
	plt.xlim([-10, 1000])
	plt.grid(True,color='white')
	plt.show()
	# plt.savefig('plots/priority/lhs-1.png')

def plot_all_samples(penalized):
	all_true_eval_dcgan = np.genfromtxt('../../outputs/dcgan/all_true_eval.dat')[3:]
	obj_min_dcgan = np.genfromtxt('../../outputs/dcgan/obj_min.dat')
	all_true_eval_lhs = np.genfromtxt('../../outputs/lhs/all_true_eval.dat')[3:]
	obj_min_lhs = np.genfromtxt('../../outputs/lhs/obj_min.dat')
	all_true_eval_mixed = np.genfromtxt('../../outputs/mixed/all_true_eval.dat')[3:]
	obj_min_mixed = np.genfromtxt('../../outputs/mixed/obj_min.dat')
	base_indiv = np.genfromtxt('../../outputs/lhs/all_true_eval.dat')[0]
	plt.axvspan(-10,100,0.0,1.0,facecolor='darkgrey', alpha=0.5)
	plt.axvspan(100,1000,0.0,1.0,facecolor='lightgrey', alpha=0.5)

	eval_counter_lhs = [i for i in range(len(all_true_eval_lhs))]
	eval_counter_dcgan = [i for i in range(len(all_true_eval_dcgan))]
	eval_counter_mixed = [i for i in range(len(all_true_eval_mixed))]

	if penalized:
		obj_lhs = np.zeros(len(all_true_eval_lhs))
		obj_dcgan = np.zeros(len(all_true_eval_dcgan))
		obj_mixed = np.zeros(len(all_true_eval_mixed))
		for indiv in range(len(all_true_eval_lhs)):
			obj_lhs[indiv] = all_true_eval_lhs[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_lhs[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval_lhs[indiv,2])))*10000
			# penalized_obj[indiv] = all_true_eval_lhs[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_lhs[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval_lhs[indiv,2])))*10000
		for indiv in range(len(all_true_eval_dcgan)):
			obj_dcgan[indiv] = all_true_eval_dcgan[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_dcgan[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval_dcgan[indiv,2])))*10000
			# penalized_obj[indiv] = all_true_eval_dcgan[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_dcgan[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval_dcgan[indiv,2])))*10000
		for indiv in range(len(all_true_eval_mixed)):
			obj_mixed[indiv] = all_true_eval_mixed[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_mixed[indiv,1])))*1000 + np.abs(np.max((0.0,all_true_eval_mixed[indiv,2])))*10000
			# penalized_obj_mixed[indiv] = all_true_eval_mixed[indiv,0]*10000 + np.abs(np.max((0.0,all_true_eval_mixed[indiv,3])))*1000

		# base = base_indiv[0]*10000 + np.abs((0.5-base_indiv[2]))*1000 + np.abs(np.max((0.0,base_indiv[3])))*1000
		base = base_indiv[0]*10000 + np.abs(np.max((0.0,base_indiv[1])))*10000 + np.abs(np.max((0.0,base_indiv[2])))*10000
	# plt.plot(obj_min[:,0],obj_min[:,1]*10000,'r-',label='Feas min obj LHS')
	# plt.plot(obj_min_dcgan[:,0],obj_min_dcgan[:,1]*10000,'b-',label='Feas min obj DCGAN')
	# plt.plot(obj_min_mixed[:,0],obj_min_mixed[:,1]*10000,'g-',label='Feas min obj MIXED')
	plt.plot(eval_counter_lhs, obj_lhs,'ro',markersize=3,label='LHS')
	plt.plot(eval_counter_dcgan, obj_dcgan,'bo',markersize=3,label='DCGAN')
	plt.plot(eval_counter_mixed[100:], obj_mixed[100:],'go',markersize=3,label='DCGAN+GF')
	plt.plot(1,base,'o',color='slategrey',label='Baseline')
	plt.hlines(base,0,1500,color='slategrey',linestyle='dashed')
	plt.xlabel('Number of CFD evaluations')
	plt.ylabel('Penalized objective (count)')
	plt.title('Penalized objective function vs No of true eval')
	plt.legend(loc='upper right')
	plt.xlim([-10, 610])
	# plt.ylim([150, 1500])
	# plt.ylim([200, 300])
	plt.grid(True,color='white')
	plt.show()
	# plt.savefig('plots/production/all_samples.svg',format='svg',dpi=300)

def plot_minimum_feasible_drag():
	"""
	Plotting history of optimization only
	"""
	# Importing outputs
	obj_min_lhs = np.genfromtxt('../../outputs/lhs/obj_min.dat')
	obj_min_dcgan = np.genfromtxt('../../outputs/dcgan/obj_min.dat')
	obj_min_mixed = np.genfromtxt('../../outputs/mixed/obj_min.dat')

	obj_min_lhs = np.concatenate((np.zeros((10,2)),obj_min_lhs))
	obj_min_dcgan = np.concatenate((np.zeros((10,2)), obj_min_dcgan))
	obj_min_mixed = np.concatenate((np.zeros((10,2)), obj_min_mixed))
	
	print(f'Feasible Optim LHS {obj_min_lhs[-1,1]}')
	print(f'Feasible Optim DCGAN {obj_min_dcgan[-1,1]}')
	print(f'Optim DCGAN+GF {obj_min_mixed[-1,1]}')
	print(f'Baseline 0.021274524750612523')
	# print(np.argmin(eval_optim_mixed[:,0]))
	base_indiv = np.genfromtxt('../../outputs/lhs/all_true_eval.dat')[0]
	ctr = np.array([i for i in range(90,100)])
	for i in range(10):
		obj_min_lhs[i,0] = i+90
		obj_min_lhs[i,1] = obj_min_lhs[10,1]
		obj_min_dcgan[i,0] = i+90
		obj_min_dcgan[i,1] = obj_min_dcgan[10,1]
		obj_min_mixed[i,0] = i+90
		obj_min_mixed[i,1] = obj_min_mixed[10,1]

	plt.plot(obj_min_lhs[:,0],obj_min_lhs[:,1],'r-',label='LHS')
	plt.plot(obj_min_dcgan[:,0],obj_min_dcgan[:,1],'b-',label='DCGAN')
	plt.plot(obj_min_mixed[:,0],obj_min_mixed[:,1],'g-',label='DCGAN+GF')
	plt.hlines(base_indiv[0]*10000,0,1500,color='slategrey',linestyle='dashed',label='Baseline')
	plt.xlabel('Number of CFD evaluations')
	plt.ylabel('Drag coefficient (count)')
	plt.title('Optimization history (minimum feasible solution)')
	plt.legend(loc='upper right')
	plt.xlim([90, 605])
	# plt.ylim([50, 250])
	plt.grid(True,color='white')
	plt.show()
	# plt.savefig('plots/production/opt_history.svg',format='svg',dpi=300)

def plot_all_solutions(method):
	all_true_eval = np.genfromtxt(f'../../outputs/{method}/all_true_eval.dat')
	all_true_eval_feas   = np.delete(all_true_eval, np.where(all_true_eval[:,-1]>0.0),  axis=0)
	all_true_eval_infeas = np.delete(all_true_eval, np.where(all_true_eval[:,-1]==0.0), axis=0)

	base_eval   = all_true_eval[0]
	min_eval    = all_true_eval[1]
	max_eval    = all_true_eval[2]
	init_points = all_true_eval[3:104]
	
	nds_idx = fast_non_dominated_sort(all_true_eval_feas[:,[0,1]])[0]
	nds_points = all_true_eval_feas[nds_idx]

	# Plotting
	fig, ax = plt.subplots()

	ax.plot(nds_points[:,1], nds_points[:,0], 'ro', label='Non-dominated designs')
	ax.plot(all_true_eval_feas[:,1],   all_true_eval_feas[:,0],   'ko', markersize=1, label='All feasible designs')
	ax.plot(all_true_eval_infeas[:,1], all_true_eval_infeas[:,0], 'kx', markersize=4, label='All infeasible designs')
	ax.plot(init_points[:,1], init_points[:,0], 'mo', fillstyle='none', label='Initial designs')

	ax.plot(base_eval[1], base_eval[0], 'bo', label='RAE2822 baseline')
	# ax.plot(min_eval[1], min_eval[0], 'ro', label='Design at dv_min')
	# ax.plot(max_eval[1], max_eval[0], 'go', label='Design at dv_max')
	# rect = Rectangle((base_eval[1],base_eval[0]), -0.9-base_eval[1], 0.012-base_eval[0], facecolor='lightgrey')
	# ax.add_patch(rect)
	plt.xlim([-1.2, 0.0])
	plt.ylim([0.005, 0.09])
	plt.xlabel('$-C_{L}$')
	plt.ylabel('$C_{D}$')
	if method=='lhs': method_title = 'LHS'
	elif method=='dcgan': method_title = 'DCGAN'
	elif method=='mixed': method_title = 'DCGAN+GF'
	plt.title(f'All solutions - {method_title}')
	plt.legend(loc='upper left')
	plt.show()

def plot_HV_performance():
	HV_lhs = np.genfromtxt('../../outputs/lhs/hv_list.dat')
	HV_dcgan = np.genfromtxt('../../outputs/dcgan/hv_list.dat')
	HV_mixed = np.genfromtxt('../../outputs/mixed/hv_list.dat')

	plt.plot(HV_lhs[1:,0],HV_lhs[1:,1],'rx-', label='LHS')
	plt.plot(HV_dcgan[1:,0],HV_dcgan[1:,1],'bx-', label='DCGAN')
	plt.plot(HV_mixed[1:,0],HV_mixed[1:,1],'gx-', label='DCGAN+GF')
	plt.xlabel('Number of designs')
	plt.ylabel('Hypervolume Value')
	plt.title('Optimization Performance - HV')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.show()

# plot_init_samples(penalized=False)
# plot_performance(method='lhs')
# plot_performance(method='dcgan')
# plot_performance(method='mixed')
# plot_all_samples(penalized=True)
# plot_minimum_feasible_drag()

# Multi-objective
plot_all_solutions(method='lhs')
plot_all_solutions(method='dcgan')
plot_all_solutions(method='mixed')

plot_HV_performance()
