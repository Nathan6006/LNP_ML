import numpy as np 
import os
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score # type: ignore
from train_multitask import train_multitask_model, get_base_args, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args, predict_multitask_from_json_cv
from rdkit import Chem # type: ignore
from rdkit.Chem import Descriptors # type: ignore
import matplotlib.pyplot as plt # type: ignore
import scipy.stats
import json
import sys
import random
import chemprop # type: ignore

# called in make_pred_vs_actual, analyze_predictions_cv, specified_cv_split
def path_if_none(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)

# these functions called in main 
def make_pred_vs_actual(split_folder, ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
	# Makes predictions on each test set in a cross-validation-split system
	# Not used for screening a new library, used for predicting on the test set of the existing dataset
	for cv in range(ensemble_size):
		data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
		results_dir = '../results/crossval_splits/'+split_folder+'/cv_'+str(cv)
		path_if_none(results_dir)
		


		output = pd.read_csv(data_dir+'/test.csv')
		metadata = pd.read_csv(data_dir+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		try:
			output = pd.read_csv(results_dir+'/predicted_vs_actual.csv')
		except:
			try:
				current_predictions = pd.read_csv(data_dir+'/preds.csv')
			except:
				arguments = [
					'--test_path',data_dir+'/test.csv',
					'--features_path',data_dir+'/test_extra_x.csv',
					'--checkpoint_dir', data_dir,
					'--preds_path',data_dir+'/preds.csv'
				]
				if 'morgan' in split_folder:
					arguments = arguments + ['--features_generator','morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				preds = chemprop.train.make_predictions(args=args)	
			# os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
				current_predictions = pd.read_csv(data_dir+'/preds.csv')
			
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
			output.to_csv(results_dir+'/predicted_vs_actual.csv', index = False)
	if '_with_ultra_held_out' in split_folder:
		results_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
		uho_dir = '../data/crossval_splits/'+split_folder+'/ultra_held_out'
		output = pd.read_csv(uho_dir+'/test.csv')
		metadata = pd.read_csv(uho_dir+'/test_metadata.csv')
		output = pd.concat([metadata, output], axis = 1)
		for cv in range(ensemble_size):
			model_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
			try:
				current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
			except:
				arguments = [
					'--test_path',uho_dir+'/test.csv',
					'--features_path',uho_dir+'/test_extra_x.csv',
					'--checkpoint_dir', model_dir,
					'--preds_path',results_dir+'/preds_cv_'+str(cv)+'.csv'
				]
				if 'morgan' in split_folder:
					arguments = arguments + ['--features_generator','morgan_count']
				args = chemprop.args.PredictArgs().parse_args(arguments)
				preds = chemprop.train.make_predictions(args=args)
				current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
			current_predictions.drop(columns = ['smiles'], inplace = True)
			for col in current_predictions.columns:
				if standardize_predictions:
					preds_to_standardize = current_predictions[col]
					std = np.std(preds_to_standardize)
					mean = np.mean(preds_to_standardize)
					current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
				current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
			output = pd.concat([output, current_predictions], axis = 1)
		pred_cols = [col for col in output.columns if '_pred_' in col]
		output['Avg_pred_quantified_delivery'] = output[pred_cols].mean(axis = 1)
		output.to_csv(results_dir+'/predicted_vs_actual.csv',index = False)

def analyze_predictions_cv(split_name,pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = '../results/crossval_splits/', ensemble_number = 5, min_values_for_analysis = 10):
	summary_table = pd.DataFrame({})
	all_names = {}
	# all_dtypes = {}
	all_ns = {}
	all_pearson = {}
	all_pearson_p_val = {}
	all_kendall = {}
	all_spearman = {}
	all_rmse = {}
	all_unique = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/predicted_vs_actual.csv')
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		all_unique = all_unique + list(set(pred_split_names))
	unique_pred_split_names = set(all_unique)
	for un in unique_pred_split_names:
		# all_names[un] = []
		# all_dtype,s[un] = []
		all_ns[un] = []
		all_pearson[un] = []
		all_pearson_p_val[un] = []
		all_kendall[un] = []
		all_spearman[un] = []
		all_rmse[un] = []
	for i in range(ensemble_number):
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/cv_'+str(i)+'/predicted_vs_actual.csv')
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		preds_vs_actual['Prediction_split_name'] = pred_split_names
		# unique_pred_split_names = set(pred_split_names)
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col[:3]=='cv_':
				data_types.append(col)
			


		for pred_split_name in unique_pred_split_names:
			path_if_none(path_to_preds+split_name+'/cv_'+str(i)+'/results')
			data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
			value_names = set(list(data_subset.Value_name))
			if len(value_names)>1:
				raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
			elif len(value_names)==0:
				value_name = 'Empty, ignore!'
			else:
				value_name = [val_name for val_name in value_names][0]
			kept_dtypes = []
			for dtype in data_types:

				analyzed_path = path_to_preds+split_name+'/cv_'+str(i)+'/results/'+pred_split_name+'/'+dtype
				path_if_none(analyzed_path)

				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				actual = data_subset['quantified_delivery']
				pred = data_subset['cv_'+str(i)+'_pred_quantified_delivery']
				if len(actual)>=min_values_for_analysis:
					pearson = scipy.stats.pearsonr(actual, pred)
					spearman, pval = scipy.stats.spearmanr(actual, pred)
					kendall, pval = scipy.stats.kendalltau(actual, pred)
				
					rmse = np.sqrt(mean_squared_error(actual, pred))
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [rmse]
					
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [pearson[0]]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [pearson[1]]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [kendall]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [spearman]
					plt.figure()
					plt.scatter(pred,actual,color = 'black')
					plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
					plt.xlabel('Predicted '+value_name)
					plt.ylabel('Experimental '+value_name)
					plt.savefig(analyzed_path+'/pred_vs_actual.png')
					plt.close()
				else:
					all_rmse[pred_split_name] = all_rmse[pred_split_name] + [float('nan')]
					all_pearson[pred_split_name] = all_pearson[pred_split_name] + [float('nan')]
					all_pearson_p_val[pred_split_name] = all_pearson_p_val[pred_split_name] + [float('nan')]
					all_kendall[pred_split_name] = all_kendall[pred_split_name] + [float('nan')]
					all_spearman[pred_split_name] = all_spearman[pred_split_name] + [float('nan')]

				all_ns[pred_split_name] = all_ns[pred_split_name] + [len(pred)]

				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
	crossval_results_path = path_to_preds+split_name+'/crossval_performance'
	path_if_none(crossval_results_path)


	pd.DataFrame.from_dict(all_ns).to_csv(crossval_results_path+'/n_vals.csv', index = True)
	pd.DataFrame.from_dict(all_pearson).to_csv(crossval_results_path+'/pearson.csv', index = True)
	pd.DataFrame.from_dict(all_pearson_p_val).to_csv(crossval_results_path+'/pearson_p_val.csv', index = True)
	pd.DataFrame.from_dict(all_kendall).to_csv(crossval_results_path+'/kendall.csv', index = True)
	pd.DataFrame.from_dict(all_spearman).to_csv(crossval_results_path+'/spearman.csv', index = True)
	pd.DataFrame.from_dict(all_rmse).to_csv(crossval_results_path+'/rmse.csv', index = True)


	# Now analyze the ultra-held-out set
	try:
		preds_vs_actual = pd.read_csv(path_to_preds+split_name+'/ultra_held_out/predicted_vs_actual.csv')
		names = []
		ns = []
		pearsons = []
		pearson_p_vals = []
		kendalls = []
		spearmans = []
		rmses = []
		split_names = []

		all_unique = []
			
		pred_split_names = []
		for index, row in preds_vs_actual.iterrows():
			pred_split_name = ''
			for vbl in pred_split_variables:
				pred_split_name = pred_split_name + row[vbl] + '_'
			pred_split_names.append(pred_split_name[:-1])
		all_unique = all_unique + list(set(pred_split_names))
		unique_pred_split_names = set(all_unique)
		preds_vs_actual['Prediction_split_name'] = pred_split_names
		# unique_pred_split_names = set(pred_split_names)
		cols = preds_vs_actual.columns
		data_types = []
		for col in cols:
			if col.startswith('Avg_pred_'):
				data_types.append(col)
			

		for pred_split_name in unique_pred_split_names:
			# path_if_none(path_to_preds+split_name+'/ultra_held_out/results')
			split_names.append(pred_split_name)
			data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name']==pred_split_name].reset_index(drop=True)
			value_names = set(list(data_subset.Value_name))
			if len(value_names)>1:
				raise Exception('Multiple types of measurement in the same prediction split: split ',pred_split_name,' has value names ',value_names,'. Try adding more pred split variables.')
			elif len(value_names)==0:
				value_name = 'Empty, ignore!'
			else:
				value_name = [val_name for val_name in value_names][0]
			kept_dtypes = []
			for dtype in data_types:
				analyzed_path = path_to_preds+split_name+'/ultra_held_out/individual_dataset_results/'+pred_split_name
				path_if_none(analyzed_path)
				kept_dtypes.append(dtype)
				analyzed_data = pd.DataFrame({'smiles':data_subset.smiles})
				analyzed_data['quantified_delivery'] = data_subset['quantified_delivery']
				analyzed_data['Avg_pred_quantified_delivery'] = data_subset['Avg_pred_quantified_delivery']
				actual = data_subset['quantified_delivery']
				pred = data_subset['Avg_pred_quantified_delivery']

				pearson = scipy.stats.pearsonr(actual, pred)
				spearman, pval = scipy.stats.spearmanr(actual, pred)
				kendall, pval = scipy.stats.kendalltau(actual, pred)

				rmse = np.sqrt(mean_squared_error(actual, pred))

				rmses.append(rmse)
				pearsons.append(pearson[0])
				pearson_p_vals.append(pearson[1])
				kendalls.append(kendall)
				spearmans.append(spearman)
				ns.append(len(pred))

				plt.figure()
				plt.scatter(pred,actual,color = 'black')
				plt.plot(np.unique(pred),np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
				plt.xlabel('Predicted '+value_name)
				plt.ylabel('Experimental '+value_name)
				plt.savefig(analyzed_path+'/pred_vs_actual.png')
				plt.close()

				analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
		uho_results_path = path_to_preds+split_name+'/ultra_held_out'
		path_if_none(uho_results_path)
		uho_results = pd.DataFrame({})
		uho_results['dataset_ID'] = split_names
		uho_results['n'] = ns
		uho_results['pearson'] = pearsons
		uho_results['pearson_p_val'] = pearson_p_vals
		uho_results['kendall'] = kendalls
		uho_results['spearman'] = spearmans
		uho_results['rmse'] = rmses


		uho_results.to_csv(uho_results_path+'/ultra_held_out_results.csv', index = False)
	except:
		pass

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'):
	# Each folder contains the following files: 
	# main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, the activity measurements for that measurement
	# If the same ionizable lipid is measured multiple times (i.e. for different properties, or transfection in vitro and in vivo) make separate rows, one for each measurement
	# formulations.csv: a csv file with columns:
		# Cationic_Lipid_Mol_Ratio
		# Phospholipid_Mol_Ratio
		# Cholesterol_Mol_Ratio
		# PEG_Lipid_mol_ratio
		# Cationic_Lipid_to_mRNA_weight_ratio
		# Helper_lipid_ID
		# If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
		# Otherwise match the row to the data in formulations.csv
	# individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
		# For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
		# Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
			# alternatively, default sample weight of 1
	# experiment_metadata.csv: contains metadata about particular dataset. This includes:
		# Experiment_ID: each experiment will be given a unique ID.
		# There will be two ROWS and any number of columns

	# Based on these files, Merge_datasets will merge all the datasets into one dataset. In particular, it will output 2 files:
		# all_merged.csv: each row  will contain all the data for a measurement (SMILES, info on dose/formulation/etc, metadata, sample weights, activity value)
		# col_type.csv: two columns, column name and type. Four types: Y_val, X_val, X_val_cat (categorical X value), Metadata, Sample_weight

	# Some metadata columns that should be held consistent, in terms of names:
		# Purity ("Pure" or "Crude")
		# ng_dose (for the dose, duh)
		# Sample_weight
		# Amine_SMILES
		# Tail_SMILES
		# Library_ID
		# Experimenter_ID
		# Experiment_ID
		# Cargo (siRNA, DNA, mRNA, RNP are probably the relevant 4 options)
		# Model_type (either the cell type or the name of the animal (probably "mouse"))


	all_df = pd.DataFrame({})
	col_type = {'Column_name':[],'Type':[]}
	experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
	if experiment_list == None:
		experiment_list = list(experiment_df.Experiment_ID)
		print(experiment_list)
	y_val_cols = []
	helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

	for folder in experiment_list:
		print(folder)
		contin = False
		try:
			main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
			contin = True
		except:
			pass
		if contin:
			y_val_cols = y_val_cols + list(main_temp.columns)
			for col in main_temp.columns:
				if 'Unnamed' in col:
					print('\n\n\nTHERE IS A BS UNNAMED COLUMN IN FOLDER: ',folder,'\n\n')
			data_n = len(main_temp)
			formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')

			try:
				individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
			except:
				individual_temp = pd.DataFrame({})
			if len(formulation_temp) == 1:
				formulation_temp = pd.concat([formulation_temp]*data_n,ignore_index = True)
			elif len(formulation_temp) != data_n:
				print(len(formulation_temp))
				to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
				raise ValueError(to_raise)

			# Change formulations from mass to molar ratio
			form_cols = formulation_temp.columns
			mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
			molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
			mass_count = 0
			molar_count = 0
			for col in form_cols:
				if col in mass_ratio_variables:
					mass_count += 1
				elif col in molar_ratio_variables:
					molar_count += 1
			if mass_count>0 and molar_count>0:
				raise ValueError('For experiment ID: ',folder,': Formulation information includes both mass and molar ratios.')
			elif mass_count<4 and molar_count<4:
				raise ValueError('For experiment ID: ',folder,': Formulation not completely specified, mass count: ',mass_count,', molar count: ',molar_count)
			elif mass_count == 4:
				cat_lip_mol_fracs = []
				phos_mol_fracs = []
				chol_mol_fracs = []
				peg_lip_mol_fracs = []
				# Change mass ratios to weight ratios
				for i in range(len(formulation_temp)):
					phos_id = formulation_temp['Helper_lipid_ID'][i]
					ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
					ion_lipid_mol_weight = Chem.Descriptors.MolWt(ion_lipid_mol)
					phospholipid_mol_weight = helper_mol_weights[phos_id][0]
					cholesterol_mol_weight = helper_mol_weights['Cholesterol']
					PEG_lipid_mol_weight = helper_mol_weights['C14-PEG2000']
					ion_lipid_moles = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
					phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
					cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
					PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
					mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
					cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
					phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
					chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
					peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
				formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
				formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
				formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
				formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

		
			if len(individual_temp) != data_n:
				print(len(individual_temp))
				raise ValueError('For experiment ID: ',folder,': Length of individual metadata file  (',len(individual_temp), ') doesn\'t match length of main datafile (',data_n,')')
			experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
			experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index = True).reset_index(drop = True)
			to_drop = []
			for col in experiment_temp.columns:
				if col in individual_temp.columns:
					print('Column ',col,' in experiment ID ',folder,'is being provided for each individual lipid.')
					to_drop.append(col)
			experiment_temp = experiment_temp.drop(columns = to_drop)
			folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis = 1).reset_index(drop = True)
			folder_df = pd.concat([folder_df, experiment_temp], axis = 1)
			# print(folder_df.columns)
			if 'Sample_weight' not in folder_df.columns:
				# print(folder)
				# folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i])/list(folder_df.smiles).count(smile) for i,smile in enumerate(folder_df.smiles)]
				folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i,smile in enumerate(folder_df.smiles)]

			all_df = pd.concat([all_df,folder_df], ignore_index = True)

	# Make the column type dict
	extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
	# ADD HELPER LIPID ID
	# extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
	extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']

	# Make changes:
	all_df = all_df.replace('im','intramuscular')
	all_df = all_df.replace('iv','intravenous')
	all_df = all_df.replace('a549','lung_epithelium')
	all_df = all_df.replace('bdmc','macrophage')
	all_df = all_df.replace('bmdm','dendritic_cell')
	all_df = all_df.replace('hela','generic_cell')
	all_df = all_df.replace('hek','generic_cell')
	all_df = all_df.replace('igrov1','generic_cell')
	all_df = all_df.replace({'Model_type':'muscle'},'Mouse')

	# other_x_vals = ['Target_organ']
	# form_variables.append('Helper_lipid_ID')

	for x_cat in extra_x_categorical:
		dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
		print(dummies.columns)
		all_df = pd.concat([all_df, dummies], axis = 1)
		extra_x_variables = extra_x_variables + list(dummies.columns)

	for column in all_df.columns:
		col_type['Column_name'].append(column)
		if column in y_val_cols:
			col_type['Type'].append('Y_val')
		elif column in extra_x_variables:
			col_type['Type'].append('X_val')
		elif column in extra_x_categorical:
			col_type['Type'].append('Metadata')
		elif column == 'Sample_weight':
			col_type['Type'].append('Sample_weight')
		else:
			col_type['Type'].append('Metadata')

	col_type_df = pd.DataFrame(col_type)
	# print(col_type_df)
	norm_split_names, norm_del = generate_normalized_data(all_df)
	all_df['split_name_for_normalization'] = norm_split_names
	all_df.rename(columns = {'quantified_delivery':'unnormalized_delivery'}, inplace = True)
	all_df['quantified_delivery'] = norm_del
	# all_df = all_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)  # Convert all strings to lower case
	all_df = all_df.replace({True: 1.0, False: 0.0})
	all_df.to_csv(write_path + '/all_data.csv', index = False)
	col_type_df.to_csv(write_path + '/col_type.csv', index = False)

def specified_cv_split(split_spec_fname, path_to_folders = '../data', is_morgan = False, cv_fold = 5, ultra_held_out_fraction = -1.0, min_unique_vals = 2.0, test_is_valid = False):
	# Splits the dataset according to the specifications in split_spec_fname
	# cv_fold: self-explanatory
	# ultra_held_out_fraction: if you want to hold a dataset out from even the cross-validation datasets this is the way to do it
	# test_is_valid: if true, then does the split where the test set is just the validation set, so that maximum data can be reserved for training set (this is for doing in siico screening)
	all_df = pd.read_csv(path_to_folders + '/all_data.csv')
	split_df = pd.read_csv(path_to_folders+'/crossval_split_specs/'+split_spec_fname)
	split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
	if ultra_held_out_fraction>-0.5:
		split_path = split_path + '_with_ultra_held_out'
	if is_morgan:
		split_path = split_path + '_morgan'
	if test_is_valid:
		split_path = split_path + '_for_in_silico_screen'
	if ultra_held_out_fraction>-0.5:
		path_if_none(split_path + '/ultra_held_out')
	for i in range(cv_fold):
		path_if_none(split_path+'/cv_'+str(i))

	perma_train = pd.DataFrame({})
	ultra_held_out = pd.DataFrame({})
	cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]

	for index, row in split_df.iterrows():
		dtypes = row['Data_types_for_component'].split(',')
		vals = row['Values'].split(',')
		df_to_concat = all_df
		for i, dtype in enumerate(dtypes):
			df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
		values_to_split = df_to_concat[row['Data_type_for_split']]
		unique_values_to_split = list(set(values_to_split))
		# print(row)
		if row['Train_or_split'].lower() == 'train' or len(unique_values_to_split)<min_unique_vals*cv_fold:
			perma_train = pd.concat([perma_train, df_to_concat])
		elif row['Train_or_split'].lower() == 'split':
			cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
			to_concat = df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]
			# print('Type: ',type(to_concat))
			# print('Ultra held out type: ',type(ultra_held_out))
			ultra_held_out = pd.concat([ultra_held_out, to_concat])
			for i, val in enumerate(cv_split_values):
				cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])

	col_types = pd.read_csv(path_to_folders + '/col_type.csv')

	# Now move the dfs to datafiles
	if ultra_held_out_fraction>-0.5:
		y,x,w,m = split_df_by_col_type(ultra_held_out,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/ultra_held_out','test')

	for i in range(cv_fold):
		test_df = cv_splits[i]
		train_inds = list(range(cv_fold))
		train_inds.remove(i)
		if test_is_valid:
			valid_df = cv_splits[i]
		else:
			valid_df = cv_splits[(i+1)%cv_fold]
			train_inds.remove((i+1)%cv_fold)
		train_df = pd.concat([perma_train]+[cv_splits[k] for k in train_inds])

		y,x,w,m = split_df_by_col_type(test_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'test')
		y,x,w,m = split_df_by_col_type(valid_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
		y,x,w,m = split_df_by_col_type(train_df,col_types)
		yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')
# these functions called in main 

# called in merge_datasets
def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
	split_names = []
	norm_dict = {}
	for index, row in all_df.iterrows():
		split_name = ''
		for vbl in split_variables:
			# print(row[vbl])
			# print(vbl)
			split_name = split_name + str(row[vbl])+'_'
		split_names.append(split_name[:-1])
	unique_split_names = set(split_names)
	for split_name in unique_split_names:
		data_subset = all_df[[spl==split_name for spl in split_names]]
		norm_dict[split_name] = (np.mean(data_subset['quantified_delivery']), np.std(data_subset['quantified_delivery']))
	norm_delivery = []
	for i, row in all_df.iterrows():
		val = row['quantified_delivery']
		split = split_names[i]
		stdev = norm_dict[split][1]
		mean = norm_dict[split][0]
		norm_delivery.append((float(val)-mean)/stdev)
	return split_names, norm_delivery

# these functions only used in specified_cv_split
def split_df_by_col_type(df,col_types):
	# Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
	y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
	x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
	# print(x_vals_cols)
	xvals_df = df[x_vals_cols]
	# print('SUCCESSFUL!!!')
	weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
	metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
	return df[y_vals_cols],xvals_df,df[weight_cols],df[metadata_cols]

def yxwm_to_csvs(y, x, w, m, path,settype):
	# y is y values
	# x is x values
	# w is weights
	# m is metadata
	# set_type is either train, valid, or test
	y.to_csv(path+'/'+settype+'.csv', index = False)
	x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
	w.to_csv(path + '/' + settype + '_weights.csv', index = False)
	m.to_csv(path + '/' + settype + '_metadata.csv', index = False)

def split_for_cv(vals,cv_fold, held_out_fraction):
	# randomly splits vals into cv_fold groups, plus held_out_fraction of vals are completely held out. So for example split_for_cv(vals,5,0.1) will hold out 10% of data and randomly put 18% into each of 5 folds
	random.shuffle(vals)
	held_out_vals = vals[:int(held_out_fraction*len(vals))]
	cv_vals = vals[int(held_out_fraction*len(vals)):]
	return [cv_vals[i::cv_fold] for i in range(cv_fold)],held_out_vals

# these functions only used in specified_cv_split

def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'train':
		split_folder = argv[2]
		epochs = 50
		cv_num = 5
		for i, arg in enumerate(argv):
			if arg.replace('–', '-') == '--epochs':
				epochs = argv[i+1]
				print('this many epochs: ',str(epochs))

		for cv in range(cv_num):
			split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
			arguments = [
				'--epochs',str(epochs),
				'--save_dir',split_dir,
				'--seed','42',
				'--dataset_type','regression',
				'--data_path',split_dir+'/train.csv',
				'--features_path', split_dir+'/train_extra_x.csv',
				'--separate_val_path', split_dir+'/valid.csv',
				'--separate_val_features_path', split_dir+'/valid_extra_x.csv',
				'--separate_test_path',split_dir+'/test.csv',
				'--separate_test_features_path',split_dir+'/test_extra_x.csv',
				'--data_weights_path',split_dir+'/train_weights.csv',
				'--config_path','../data/args_files/optimized_configs.json',
				'--loss_function','mse','--metric','rmse'
			]
			if 'morgan' in split_folder:
				arguments += ['--features_generator','morgan_count']
			args = chemprop.args.TrainArgs().parse_args(arguments)
			mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
			print("mean score:", mean_score)
			print("std_store:", std_score)

	elif task_type == 'predict':
		cv_num = 5
		split_model_folder = '../data/crossval_splits/'+argv[2]
		screen_name = argv[3]
		# READ THE METADATA FILE TO A DF, THEN TAG ON THE PREDICTIONS TO GENERATE A COMPLETE PREDICTIONS FILE
		all_df = pd.read_csv('../data/libraries/'+screen_name+'/'+screen_name+'_metadata.csv')
		for cv in range(cv_num):
			# results_dir = '../results/crossval_splits/'+split_model_folder+'cv_'+str(cv)
			arguments = [
				'--test_path','../data/libraries/'+screen_name+'/'+screen_name+'.csv',
				'--features_path','../data/libraries/'+screen_name+'/'+screen_name+'_extra_x.csv',
				'--checkpoint_dir', split_model_folder+'/cv_'+str(cv),
				'--preds_path','../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv'
			]
			if 'morgan' in split_model_folder:
					arguments = arguments + ['--features_generator','morgan_count']
			args = chemprop.args.PredictArgs().parse_args(arguments)
			preds = chemprop.train.make_predictions(args=args)
			new_df = pd.read_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv')
			all_df['smiles'] = new_df.smiles
			all_df['cv_'+str(cv)+'_pred_delivery'] = new_df.quantified_delivery	
		all_df['avg_pred_delivery'] = all_df[['cv_'+str(cv)+'_pred_delivery' for cv in range(cv_num)]].mean(axis=1)
		all_df.to_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/pred_file.csv', index = False)
	elif task_type == 'hyperparam_optimize':
		split_folder = argv[2]
		data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
		arguments = [
			'--data_path',data_dir+'/train.csv',
			'--features_path', data_dir+'/train_extra_x.csv',
			'--separate_val_path', data_dir+'/valid.csv',
			'--separate_val_features_path', data_dir+'/valid_extra_x.csv',
			'--separate_test_path',data_dir+'/test.csv',
			'--separate_test_features_path',data_dir+'/test_extra_x.csv',
			'--dataset_type', 'regression',
			'--num_iters', '5',
			'--config_save_path','..results/'+split_folder+'/hyp_cv_0.json',
			'--epochs', '5'
		]
		args = chemprop.args.HyperoptArgs().parse_args(arguments)
		chemprop.hyperparameter_optimization.hyperopt(args)
	elif task_type == 'analyze':
		# output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
		split = argv[2]
		make_pred_vs_actual(split, predictions_done = [], ensemble_size = 5)
		analyze_predictions_cv(split)
	elif task_type == 'merge_datasets':
		merge_datasets(None)
	elif task_type == 'split':
		split = argv[2]
		ultra_held_out = float(argv[3])
		is_morgan = False
		in_silico_screen = False
		if len(argv)>4:
			if argv[4]=='morgan':
				is_morgan = True
				if len(argv)>5 and argv[5]=='in_silico_screen_split':
					in_silico_screen = True
			elif argv[4]=='in_silico_screen_split':
				in_silico_screen = True
		specified_cv_split(split,ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)

if __name__ == '__main__':
	main(sys.argv)


