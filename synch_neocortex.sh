rsync -aP --exclude 'data' --exclude 'scripts/deprecated' \
--exclude={'.git','gitignore','results/*'} \
~/projects/cifar/ gc349@neocortex:~/projects/cifar/