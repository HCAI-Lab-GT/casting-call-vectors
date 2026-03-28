Initially ran on single node (slurm node 1).
Paused at architect and split into 3 batches:
- role_list_1.json
- role_list_2.json
- role_list_3.json

for slurm node 1, node 2, node 3 respectively

After slurm node 1 finished early.

Removed last 10 roles from list 2 and list 3 to put into slurm node 1.

Indicated from 
- role_list_1_from_2.json
- role_list_1_from_3.json
taking from the last roles in list_2 and list_3 respectively.

This is done to ensure maximum gpu usage at all times.