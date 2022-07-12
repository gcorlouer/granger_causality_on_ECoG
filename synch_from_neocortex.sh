rsync -aP --exclude 'data' --exclude 'scripts/deprecated' \
--exclude={'.git','gitignore','results/*'} \
gc349@neocortex:~/projects/cifar/ ~/projects/cifar/