source /home/leo/.python-envs/torch-gpu-env/bin/activate
task_to_skip=3

tasks_list=(0 1 2 3)
# remove task to skip from the tasks list
tasks_list=(${tasks_list[@]/$task_to_skip})
echo tasks_list: ${tasks_list[@]}

for i in "${tasks_list[@]}"
do
    python run_clf_onDiffTask.py -c=configs/cross_clf.yaml --task_to_train=$i --task_to_skip=$task_to_skip
done
