rsync -aP --exclude 'data' --exclude 'scripts/deprecated' \
--exclude={'.git','gitignore','electrode_mapping'} \
~/projects/cifar/ gc349@neocortex:~/projects/cifar/