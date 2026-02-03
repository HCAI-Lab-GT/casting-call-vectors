---
created: 2026-02-02T23:53:53 (UTC -08:00)
tags: []
source: https://matsprogram.notion.site/MATS-Cluster-Handbook-1dfa0e14b0ad80d7a7e0dcc75927becf
author:
---

# MATS Cluster Handbook

> ## Excerpt
> A tool that connects everyday work into one space. It gives you and your teams AI tools—search, writing, note-taking—inside an all-in-one, flexible workspace.

---
### Getting Started with the MATS Cluster

Welcome to the MATS compute cluster! This guide provides the essential information you need to log in and start running jobs.

#### 1\. Getting Access

To use the cluster, you first need a Fellow account. Fill out the [Compute Request Form](https://forms.matsprogram.org/compute-request) \[7.1 fellows use [this link](https://airtable.com/appZq2f1sM0tW9kH7/shrcHXIdlDjJ8eY7E)\], select MATS Compute Cluster from Service ($0), and paste your public SSH key into the Public SSH Key field.

Note: this applies to all existing users as well as those who submitted the form recently.

We will then create your Fellow account and provide you with:

Information about which compute resources (Slurm account) you are assigned to.

#### 2\. Logging In

You will access the cluster by connecting to the Controller node via SSH Key. The controller node is where you will submit jobs. Please avoid any heavy computing on the Controller node.

Command: Open a terminal on your local machine and run:

ssh -i path/to/your/key username@94.156.8.208

If you created a passphrase (which we strongly recommend!): enter it here

#### 3\. Storage: Where Your Files Live

All storage is provided via a shared Network File System (NFS) mounted at 

/mnt/nw

. This means your files are accessible from the controller and any compute node your jobs run on.

Home directory (

/mnt/nw/home/your\_username

):

This is your private space. You land here when you log in (

pwd

 will show this path).

Use it for your personal files, configuration files (like 

.bashrc

), source code, and small datasets.

Job output files will often be written here by default unless you specify otherwise.

Only you (and system administrators) can access this directory.

Team directory (

/mnt/nw/teams/your\_team\_name

):

You will also have access to a shared team/stream directory (e.g., 

/mnt/nw/teams/team\_andrej)

.

Use this space for collaboration: shared code, common datasets, group results.

Files created here are typically accessible and modifiable by all members of your stream. Ask Compute Team if you are unsure of your team directory path.

#### 4\. Compute Resources: Partitions, QoS, & GPUs

The cluster uses the Slurm workload manager. Jobs are submitted to partitions, which are logical groupings of compute nodes, and run under a specific Quality of Service (QoS) level, which defines limits and priority. Your account will be authorized to use specific partitions and QoS levels.

Note: These initial QoS limits are a starting point. We'll monitor usage and gather feedback to adjust settings that balance research needs with fair access.

QoS levels help manage resources fairly and provide different service tiers. You have access to the following:

Checking Node Status: You can see the status of nodes and partitions using the 

sinfo

 command.

A useful format showing Partition, NodeName, GPU counts/types, and State (idle, alloc, down, etc.):

sinfo -o "%15P %15N %10G %T"

#### 5\. Running Jobs with Slurm

You submit jobs using batch scripts via the 

sbatch

 command.

Example Batch script (my\_gpu\_job.sh): Create this file in your home directory.

Submitting the Job:

sbatch my\_gpu\_job.sh

Slurm will respond with 

Submitted batch job <jobID>

.

Checking Job Status: Use the 

squeue

 command.

Look at the ST (State) column: PD (Pending), R (Running), CG (Completing).

squeue                 # Show all jobs in the queue squeue \-u your\_username # Show only your jobs



Output: Once the job finishes, the output file (

my\_job\_output\_<jobID>.log

 in the example) will appear in the directory where you ran sbatch (unless you specified a different path in 

\--output

).

Other Commands: 

srun

 (run interactive jobs), 

salloc

 (allocate resources for interactive use), 

scancel <jobID>

 (cancel a job).

#### More examples

#### 6\. Resource Usage Etiquette & Limits (WIP)

The cluster is a shared resource. Slurm uses Quality of Service (QoS) levels to manage resources fairly and enforce limits. Responsible usage is crucial. Please adhere to the following guidelines:

Understand QoS limits: Be aware of the limits associated with the QoS you are using (

normal

or

debug

). Key limits include:

Max Walltime:

normal

(7 days),

debug

(1 hour). Max GPUs per User (concurrently):

normal

(4 GPUs),

debug

(1 GPU). Jobs requesting resources beyond these limits will be rejected at submission.

Request only what you need: Be mindful when specifying resources (

#SBATCH

directives). Don't request 4 GPUs with

\-qos=normal

if your code only uses 1. Requesting fewer resources leaves more available for others and fits within tighter limits like the

debug

QoS.

Set realistic time limits: Always set a reasonable

#SBATCH --time

limit based on your job's expected duration, even within the QoS maximum. This helps the scheduler and prevents runaway jobs from holding resources unnecessarily if something goes wrong. Use the format

D-HH:MM:SS

or

HH:MM:SS

.

Use

debug

QoS appropriately: Use

\-qos=debug

for short tests, debugging, or interactive sessions where quick turnaround is needed and the 1-hour / 1-GPU limits are sufficient. Use the default

normal

QoS for longer runs or jobs needing more than 1 GPU.

Monitor your jobs: Use

squeue -u your\_username

to keep an eye on your running and pending jobs. Cancel jobs that are no longer needed (

scancel <jobID>

). Check

sacct -u your\_username

to review completed jobs and resource usage.

Communicate: If you anticipate needing resources close to the user limits (e.g., running multiple jobs concurrently using 4 GPUs under

normal

QoS) for an extended period, consider informing the admin team (message Compute Team) beforehand.

Memory Guidelines: Each L40 GPU has ~46GB VRAM. For GPU workloads, consider if you really need more than 64GB system RAM per job. Over-requesting blocks other jobs even when the memory goes unused.

Adhering to these guidelines and understanding the QoS limits ensures fair access for all users.

#### 7\. Getting Help

If you encounter issues or have questions, please reach out to Compute Team or post in the [#support-mats-cluster](https://mats-program.slack.com/archives/C08PHV9E12R) channel on the MATS Slack.
