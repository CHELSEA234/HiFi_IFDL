'''Image-level result parser file.'''
import numpy as np
import csv
import os
import argparse

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from glob import glob
from pathlib import Path

def compute_stats(gt_list, score_list, pred_list, descr=''):
	'''
		compute the AUC and macro-F1.
		descr indicates the image-editing and 
	'''
	scr_auc = roc_auc_score(gt_list, score_list)
	F1 = metrics.f1_score(gt_list, pred_list, average='macro')
	print(f"{descr}, the auc is: {scr_auc:.3f}.")
	print(f"{descr}, the F1 is: {F1:.3f}")

def main(args):
	csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	csv_file_name = os.path.join(csv_dir, args.csv_file_name)
	csv_file      = open(csv_file_name)
	csv_reader    = csv.reader(csv_file, delimiter=',')
	line_count = 0
	cnn_gt_list, cnn_pred_list, cnn_score_list = [], [], []
	edit_gt_lst, edit_pred_lst, edit_score_lst = [], [], []
	real_gt_lst, real_pred_lst, real_score_lst = [], [], []

	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		else:
			img, sc_3, pred_3, gt_3, sc_2, pred_2, gt_2, sc_1, pred_1, gt_1, sc_0, pred_0, gt_0 = row
			sc_3, pred_3, gt_3 = float(sc_3), float(pred_3), float(gt_3)

			## turn into the 0 and 1 binary value.
			res_gt_3 = 0 if gt_3 == 0 else 1
			res_pred_3 = 0 if pred_3 == 0 else 1

			if gt_3 in [4,5,6,7,8,9,10,11,12,13,14]:	# CNN-syn.
				cnn_gt_list.append(res_gt_3)
				cnn_pred_list.append(res_pred_3)
				cnn_score_list.append(sc_3)
			elif gt_3 in [1,2,3]:		# Image-editing.
				edit_gt_lst.append(res_gt_3)
				edit_pred_lst.append(res_pred_3)
				edit_score_lst.append(sc_3)
			elif gt_3 == 0:				# Real image
				real_gt_lst.append(res_gt_3)
				real_pred_lst.append(res_pred_3)
				real_score_lst.append(sc_3)
			else:
				raise ValueError('Not Found.')
			line_count += 1

	csv_file.close()

	compute_stats(real_gt_lst+cnn_gt_list, real_score_lst+cnn_score_list, real_pred_lst+cnn_pred_list, 'CNN')
	compute_stats(real_gt_lst+edit_gt_lst, real_score_lst+edit_score_lst, real_pred_lst+edit_pred_lst, 'Edit')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--csv_file_name', type=str, default='result_0.0003.csv')
	args = parser.parse_args()
	main(args)
