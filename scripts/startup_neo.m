%Startup stuff
% Run script in code_matlab rep inside CIFAR project directroy
global CIFAR_version;
CIFAR_version.major = 1;
CIFAR_version.minor = 0;

rootdir = getenv('USERPROFILE');
%fprintf('[CIFAR startup] Initialising CIFAR version %d.%d\n', PsyMEG_version.major, PsyMEG_version.minor);

% Home dir

global home_dir code_root % parent folder of CIFAR directory
home_dir = fullfile('~');
code_root = fullfile(home_dir, 'cifar','code_matlab') ; 
addpath(genpath(code_root));
rmpath(fullfile(code_root, 'deprecated')); % remove deprecated code to avoid confusion
fprintf('[CIFAR startup] Added path %s and appropriate subpaths\n',code_root);

% Initialize mvgc library

global mvgc_root;
mvgc_root = fullfile(home_dir, 'MVGC2'); %or add MVGC_deprecated
assert(exist(mvgc_root,'dir') == 7,'bad MVGC path: ''%s'' does not exist or is not a directory',mvgc_root);
cd(mvgc_root);
startup;
cd(code_root);


fprintf('[CIFAR startup] Initialised (you may re-run `startup'' at any time)\n');
