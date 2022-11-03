rsync -aP --exclude 'data/CIFAR_RECALL_DATA' --exclude 'scripts/deprecated' \
--exclude 'data/source_data' --exclude={'.git','gitignore','electrode_mapping'} \
~/projects/cifar/ gc349@neocortex:~/projects/cifar/