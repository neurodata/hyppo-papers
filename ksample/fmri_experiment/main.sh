python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --n-perms 1000 --k-sample 6 --multiway
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --n-perms 10000 --k-sample 3N
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --n-perms 10000 --k-sample 3E
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --n-perms 10000
python3 ./make_pvalue_heatmap.py --source ./dcorr_fmri_pvalues --save ./pvalues_heatmap.pdf
